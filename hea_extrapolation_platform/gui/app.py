"""
Gradio Dashboard for Extrapolation Discovery Platform
Gradio dashboard

Launch::

    python -m hea_extrapolation_platform gui --port 7860

Tab-based layout:
  1. Dashboard     - KPI cards + validity ranking + performance heatmap
  2. Data Summary  - Dataset statistics, composition/target plots, CSV upload
  3. Config & Run  - Parameter UI + run button + progress bar + streaming log
  4. Results       - Run results table + filters + parity plot
  5. OOD Map       - Interactive PCA scatter + OOD candidates table
  6. Literature    - Query UI + filters + feature frequency
  7. Report        - Markdown preview + download

Design decisions (GUI review fixes):
  - gr.State replaces module-global _SESSION for multi-user isolation (#1)
  - Slider value cast to int to avoid TypeError (#2)
  - OOD Map uses actual split indices from runner (#3)
  - InputScope filter choices match schema enum values (#4)
  - theme= passed to gr.Blocks(), not launch() (#5)
  - app.queue() enables async execution (#6)
  - run_experiment is a generator that yields progress (#7)
  - Experiment completion auto-refreshes all tabs (#8)
  - Literature index is built once and cached in state (#9)
  - Parity plot de-duplicates per unique sample index (#10)
  - Filter choices are generated dynamically from run data (#11)
  - Output directory uses timestamp-based subdirectories (#12)
"""

from __future__ import annotations

import datetime
import gc
import html as html_mod
import logging
import queue
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from hea_extrapolation_platform.gui.plotly_charts import (
    build_summary_stats_md,
    plotly_composition_heatmap,
    plotly_feature_correlation,
    plotly_feature_frequency,
    plotly_fs_grouped_bar,
    plotly_heatmap,
    plotly_ood_map,
    plotly_parity,
    plotly_target_histogram,
    plotly_uncertainty_ood,
    plotly_validity_ranking,
    runs_to_dataframe,
    validity_scores_to_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session helpers (gr.State-backed, per-user)  -- Fix #1
# ---------------------------------------------------------------------------

def _empty_session() -> Dict[str, Any]:
    """Return a fresh session dict (one per browser tab)."""
    return {
        "runs": [],
        "validity_scores": [],
        "ood_results": {},
        "ood_split_indices": {},
        "compositions_df": None,
        "features_df": None,
        "target": None,
        "runner": None,
        "report_path": None,
        "literature_engine": None,
        "literature_results": None,
        "feature_recommendation": None,
        "feature_counts": None,
        "feature_selection_results": {},  # {fs_name: FeatureSelectionSummary}
    }


# ---------------------------------------------------------------------------
# Literature index (lazy, cached per session) -- Fix #9
# ---------------------------------------------------------------------------

def _get_literature_engine(session: Dict[str, Any]) -> Any:
    """Build or return the cached literature search engine."""
    if session.get("literature_engine") is not None:
        return session["literature_engine"]

    from hea_extrapolation_platform.literature_graph.seed_data import (
        get_seed_papers, get_seed_workflows,
    )
    from hea_extrapolation_platform.literature_graph.workflow_text import (
        generate_workflow_text,
    )
    from hea_extrapolation_platform.literature_graph.vector_index import (
        build_index,
    )
    from hea_extrapolation_platform.literature_graph.search import (
        LiteratureSearchEngine,
    )

    papers = get_seed_papers()
    workflows = get_seed_workflows()
    wf_texts = [generate_workflow_text(w) for w in workflows]
    wf_ids = [w.workflow_id for w in workflows]
    index = build_index(wf_ids, wf_texts, use_faiss=True)

    engine = LiteratureSearchEngine(
        index=index, workflows=workflows, papers=papers,
    )
    session["literature_engine"] = engine
    return engine


# ---------------------------------------------------------------------------
# Integration status helper
# ---------------------------------------------------------------------------

def _build_integration_status_md(
    runner: Optional[Any],
) -> str:
    """Build a Markdown summary showing integration status.

    This is displayed inside a collapsible Accordion so that
    casual users never need to see it.  Power users can expand
    the panel to inspect backend connectivity.
    """
    from hea_extrapolation_platform.integrations.mlflow_tracker import (
        is_mlflow_available,
    )
    from hea_extrapolation_platform.integrations.feast_store import (
        is_feast_available,
    )

    mlflow_pkg = "Installed" if is_mlflow_available() else "Fallback"
    feast_pkg = "Installed" if is_feast_available() else "Fallback"

    if runner is None:
        return (
            "Experiment not yet executed.\n\n"
            "All integrations are **automatically enabled** when you "
            "run an experiment.  No configuration required."
        )

    # MLflow
    if runner.tracker.is_mlflow_active:
        mlflow_icon = "Active"
        mlflow_detail = f"Tracking URI: {runner.tracker.get_tracking_uri()}"
    else:
        tracked = runner.tracker.list_runs()
        mlflow_icon = "Active (in-memory)"
        mlflow_detail = f"{len(tracked)} run(s) tracked"

    # Feast
    if runner.feature_store.is_feast_active:
        store_info = runner.feature_store.get_store_info()
        feast_icon = "Active"
        feast_detail = f"Repo: {store_info.get('repo_path', 'N/A')}"
    else:
        sets = runner.feature_store.list_feature_sets()
        feast_icon = "Active (built-in)"
        feast_detail = f"{len(sets)} feature set(s) managed"

    # MInt
    if runner.mint_registry is not None:
        wfs = runner.mint_registry.list_workflows()
        mint_icon = f"Active ({len(wfs)} workflows)"
        names = ", ".join(w["name"] for w in wfs)
        mint_detail = names
    else:
        mint_icon = "Standby"
        mint_detail = "No MInt server connected; using built-in workflows"

    lines = [
        "### Data Pipeline",
        "",
        "```",
        "Feast (Feature Store)  -->  Experiment Runner  -->  MLflow (Tracking)",
        "                                   |                       ",
        "                            MInt (Workflows)               ",
        "```",
        "",
        "| Component | Mode | Details |",
        "|---|---|---|",
        f"| **MLflow** -- Experiment Tracking | {mlflow_icon} ({mlflow_pkg}) | {mlflow_detail} |",
        f"| **Feast** -- Feature Store | {feast_icon} ({feast_pkg}) | {feast_detail} |",
        f"| **MInt** -- Workflow Adapters | {mint_icon} | {mint_detail} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers for cross-tab refresh  -- Fix #8
# ---------------------------------------------------------------------------

def _refresh_dashboard_data(
    metric: str, session: Dict[str, Any],
) -> Tuple[str, str, str, str, str, Any, Any]:
    runs = session.get("runs", [])
    scores = session.get("validity_scores", [])
    ood_results = session.get("ood_results", {})
    runner = session.get("runner")

    n_runs = str(len(runs))
    best_fs = scores[0].feature_set if scores else "--"
    best_score = f"{scores[0].total:.4f}" if scores else "--"

    total_ood = (
        sum(r.n_ood for r in ood_results.values()) if ood_results else 0
    )
    ood_str = str(total_ood) if ood_results else "--"

    integration_md = _build_integration_status_md(runner)

    validity_fig = plotly_validity_ranking(scores) if scores else None
    heatmap_fig = plotly_heatmap(runs, metric=metric) if runs else None

    return (
        n_runs, best_fs, best_score, ood_str,
        integration_md, validity_fig, heatmap_fig,
    )


def _build_physical_interpretation_md(
    runs: List,
    scores: List,
    ood_results: Dict[str, Any],
) -> str:
    """Build Markdown with physical interpretation of analysis results."""
    if not runs or not scores:
        return (
            "*解析を実行すると、ここに物理的考察と "
            "FS比較サマリーが表示されます。*"
        )

    from hea_extrapolation_platform.feature_selection import FS_PHYSICAL_ORIGINS

    lines: List[str] = []

    # --- FS Comparison Table ---
    lines.append("### Feature Set 比較サマリー\n")

    fs_data: Dict[str, List] = {}
    for r in runs:
        fs_data.setdefault(r.feature_set, []).append(r)

    score_map = {s.feature_set: s for s in scores}
    best_fs = scores[0].feature_set if scores else ""

    lines.append(
        "| FS | 特徴量数 | RMSE(Test) Mean | R$^2$(Test) Mean "
        "| Validity Total | 推奨 |"
    )
    lines.append("|---|---|---|---|---|---|")

    _fs_sizes = {
        "FS_BASE": 8, "FS_THERMO": 11, "FS_SIZE": 12,
        "FS_ELECTRON": 11, "FS_ALL": 18, "FS_MAGPIE": 132,
    }

    for fs_name in sorted(fs_data.keys()):
        fs_runs = fs_data[fs_name]
        rmses = [r.rmse_test for r in fs_runs if r.rmse_test > 0]
        r2s = [r.r2_test for r in fs_runs]
        n_feat = _fs_sizes.get(fs_name, "?")
        rmse_mean = f"{np.mean(rmses):.2f}" if rmses else "N/A"
        r2_mean = f"{np.mean(r2s):.4f}" if r2s else "N/A"
        vs = score_map.get(fs_name)
        total = f"{vs.total:.4f}" if vs else "N/A"
        recommend = "**Best**" if fs_name == best_fs else ""
        lines.append(
            f"| {fs_name} | {n_feat} | {rmse_mean} "
            f"| {r2_mean} | {total} | {recommend} |"
        )

    lines.append("")

    # --- Best FS recommendation ---
    if scores:
        best = scores[0]
        lines.append(f"### 推奨特徴量セット: **{best.feature_set}**\n")
        lines.append(
            f"- 総合妥当性スコア: {best.total:.4f}\n"
            f"- 効果量: {best.effect_size:.4f} / "
            f"安定性: {best.stability:.4f} / "
            f"汎化性: {best.generalisation:.4f}\n"
            f"- 外挿安全性: {best.extrapolation_safety:.4f} / "
            f"リークペナルティ: {best.leak_penalty:.4f}"
        )
        lines.append("")

    # --- Physical Interpretation ---
    lines.append("### 物理的考察\n")

    if scores and len(scores) >= 2:
        best = scores[0]
        second = scores[1]
        origin_best = FS_PHYSICAL_ORIGINS.get(best.feature_set, "")
        origin_second = FS_PHYSICAL_ORIGINS.get(second.feature_set, "")

        lines.append(
            f"**{best.feature_set}** が最も高い妥当性スコア "
            f"({best.total:.4f}) を示しました。"
        )
        if origin_best:
            lines.append(f"> {origin_best}\n")

        delta = best.total - second.total
        if delta < 0.05:
            lines.append(
                f"2位の **{second.feature_set}** "
                f"({second.total:.4f}) との差は小さく、"
                "両方の特徴量セットが同等に有効である可能性があります。"
            )
        else:
            lines.append(
                f"2位の **{second.feature_set}** "
                f"({second.total:.4f}) と比較して "
                f"{delta:.4f} の差があり、"
                f"**{best.feature_set}** が明確に優位です。"
            )
        lines.append("")

        # Interpret based on which FS is best
        if "THERMO" in best.feature_set:
            lines.append(
                "**解釈**: 熱力学的安定性指標（Omega, H$_{mix}$, "
                "ss\\_formation）が予測に支配的であり、"
                "固溶体の安定性が目的変数を強く支配していることを示唆します。"
            )
        elif "ELECTRON" in best.feature_set:
            lines.append(
                "**解釈**: 電子構造プロキシ（d電子数、VEC×電気陰性度）が"
                "予測に有効であり、電子バンド構造が物性を支配していることを示唆します。"
            )
        elif "SIZE" in best.feature_set:
            lines.append(
                "**解釈**: 原子サイズ・弾性不整合指標が予測に有効であり、"
                "格子歪みによる固溶体強化が支配的メカニズムであることを示唆します。"
            )
        elif "MAGPIE" in best.feature_set:
            lines.append(
                "**解釈**: 132個のMAGPIE記述子が最も有効であり、"
                "高次元の組成情報が予測に寄与しています。"
                "データ量が十分な場合に有効ですが、過学習リスクに注意してください。"
            )
        elif "ALL" in best.feature_set:
            lines.append(
                "**解釈**: ドメイン固有の16特徴量の組み合わせが最も有効であり、"
                "複数の物理メカニズム（格子歪み、熱力学、電子構造）の"
                "相互作用が目的変数を決定していることを示唆します。"
            )
        elif "BASE" in best.feature_set:
            lines.append(
                "**解釈**: 基本記述子（delta\\_r, VEC, S$_{mix}$）のみで"
                "十分な予測精度が得られ、Hume-Rothery則に基づく"
                "単純なモデルが有効であることを示唆します。"
            )
        lines.append("")

    # --- OOD interpretation ---
    if ood_results:
        lines.append("### OOD検出サマリー\n")
        for fs_key, ood_res in sorted(ood_results.items()):
            ratio_pct = ood_res.ood_ratio * 100
            lines.append(
                f"- **{fs_key}**: {ood_res.n_ood}/{ood_res.n_total} "
                f"({ratio_pct:.1f}%) がOOD判定"
            )
        lines.append("")
        # Interpretation
        avg_ratio = np.mean(
            [r.ood_ratio for r in ood_results.values()]
        )
        if avg_ratio > 0.2:
            lines.append(
                "**注意**: OOD比率が20%超です。データの分布がトレーニング領域から"
                "大きく外れている可能性があり、外挿予測の信頼性に注意が必要です。"
            )
        elif avg_ratio > 0.1:
            lines.append(
                "OOD比率10-20%: 一部のサンプルが訓練データの分布外にあります。"
                "これらのサンプルの予測値は信頼性が低い可能性があります。"
            )
        else:
            lines.append(
                "OOD比率10%未満: テストデータの大部分が訓練データの"
                "分布内にあり、予測は比較的信頼できます。"
            )

    # --- Judgment guide ---
    lines.append("\n### 結果の読み方ガイド\n")
    lines.append(
        "| 指標 | 高い場合の解釈 | 低い場合の解釈 |\n"
        "|---|---|---|\n"
        "| RMSE(Test) | 予測精度が低い（悪い） | 予測精度が高い（良い） |\n"
        "| R$^2$(Test) | 目的変数の分散をよく説明 | 予測力が不十分 |\n"
        "| Effect Size | 特徴量セットの違いによる性能差が大きい | 差が小さい |\n"
        "| Stability | シード間での結果が安定 | 結果のばらつきが大きい |\n"
        "| Generalisation | 分割方法に依らず汎化 | 特定分割でのみ好成績 |\n"
        "| Extrap. Safety | OOD領域でも精度を維持 | 外挿で精度低下 |\n"
        "| Leak Penalty | データリークの疑い（要注意） | リーク無し |"
    )

    return "\n".join(lines)


def _refresh_results_data(
    wf_filter: str, fs_filter: str, sp_filter: str,
    session: Dict[str, Any],
) -> Tuple:
    runs = session.get("runs", [])
    scores = session.get("validity_scores", [])
    ood_results = session.get("ood_results", {})

    # Fix #11: dynamic filter choices from actual run data
    wf_choices = ["All"] + sorted({r.workflow for r in runs})
    fs_choices = ["All"] + sorted({r.feature_set for r in runs})
    sp_choices = ["All"] + sorted({r.split_policy for r in runs})

    v_df = validity_scores_to_dataframe(scores) if scores else pd.DataFrame()

    filtered = runs
    if wf_filter != "All":
        filtered = [r for r in filtered if r.workflow == wf_filter]
    if fs_filter != "All":
        filtered = [r for r in filtered if r.feature_set == fs_filter]
    if sp_filter != "All":
        filtered = [r for r in filtered if r.split_policy == sp_filter]

    r_df = runs_to_dataframe(filtered) if filtered else pd.DataFrame()
    parity_fig = plotly_parity(filtered) if filtered else None

    # Physical interpretation + FS comparison
    interp_md = _build_physical_interpretation_md(runs, scores, ood_results)

    # FS comparison grouped bar chart
    fs_bar_fig = plotly_fs_grouped_bar(runs) if runs else None

    return (
        gr.update(choices=wf_choices, value=wf_filter if wf_filter in wf_choices else "All"),
        gr.update(choices=fs_choices, value=fs_filter if fs_filter in fs_choices else "All"),
        gr.update(choices=sp_choices, value=sp_filter if sp_filter in sp_choices else "All"),
        v_df, r_df, parity_fig, interp_md, fs_bar_fig,
    )


def _refresh_ood_data(
    fs_key: str, session: Dict[str, Any],
) -> Tuple:
    ood_results = session.get("ood_results", {})
    features_df = session.get("features_df")
    comps_df = session.get("compositions_df")
    ood_split_indices = session.get("ood_split_indices", {})

    if not ood_results or features_df is None:
        return None, "No OOD results. Run experiment first.", pd.DataFrame()

    ood_res = ood_results.get(fs_key)
    if ood_res is None:
        available = list(ood_results.keys())
        return (
            None,
            f"No OOD for {fs_key}. Available: {available}",
            pd.DataFrame(),
        )

    from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
    try:
        fs_enum = FeatureSetName(fs_key)
        cols = FeatureCatalog.columns(fs_enum)
    except (ValueError, KeyError):
        return None, f"Unknown feature set: {fs_key}", pd.DataFrame()

    # Fix #3: use stored train/test indices for correct visualization
    split = ood_split_indices.get(fs_key)
    if split is not None:
        train_idx, test_idx = split
        X_train = features_df[cols].iloc[train_idx]
        X_query = features_df[cols].iloc[test_idx]
    else:
        logger.warning("No OOD split indices for %s, using heuristic", fs_key)
        X_all = features_df[cols]
        n_ood = len(ood_res.composite_scores)
        X_train = X_all.iloc[:len(X_all) - n_ood]
        X_query = X_all.iloc[len(X_all) - n_ood:]

    fig = plotly_ood_map(
        X_train, X_query,
        composite_scores=ood_res.composite_scores,
        is_ood=ood_res.is_ood,
        title=f"OOD Map (PCA) -- {fs_key}",
    )

    summary = (
        f"Total query: {ood_res.n_total} | "
        f"OOD: {ood_res.n_ood} ({ood_res.ood_ratio:.1%}) | "
        f"Threshold: {ood_res.ood_threshold:.4f}"
    )

    cand_df = pd.DataFrame()
    if comps_df is not None and ood_res.is_ood.any() and split is not None:
        _, test_idx_arr = split
        ood_mask = ood_res.is_ood
        ood_local = np.where(ood_mask)[0]
        ood_global = np.asarray(test_idx_arr)[ood_local]
        ood_global = ood_global[ood_global < len(comps_df)]
        if len(ood_global) > 0:
            cand_df = comps_df.iloc[ood_global].copy()
            cand_df["OOD_Score"] = ood_res.composite_scores[
                ood_local[:len(ood_global)]
            ]
            cand_df = cand_df.sort_values(
                "OOD_Score", ascending=False,
            ).head(20)
            cand_df = cand_df.round(3)

    return fig, summary, cand_df


def _export_ood_csv(
    fs_key: str, session: Dict[str, Any],
) -> Any:
    """Export OOD-flagged data to a temporary CSV and return a gr.update.

    Builds a DataFrame that includes:
      - composition columns from the original dataset
      - feature values for the selected feature set
      - OOD composite score and boolean flag

    Returns ``gr.update(value=path, visible=True)`` on success,
    or ``gr.update(value=None, visible=False)`` when no data is available.
    """
    import tempfile

    ood_results = session.get("ood_results", {})
    features_df = session.get("features_df")
    comps_df = session.get("compositions_df")
    target = session.get("target")
    ood_split_indices = session.get("ood_split_indices", {})

    if not ood_results or features_df is None:
        return gr.update(value=None, visible=False)

    ood_res = ood_results.get(fs_key)
    if ood_res is None:
        return gr.update(value=None, visible=False)

    from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
    try:
        fs_enum = FeatureSetName(fs_key)
        cols = FeatureCatalog.columns(fs_enum)
    except (ValueError, KeyError):
        return gr.update(value=None, visible=False)

    split = ood_split_indices.get(fs_key)
    if split is None:
        return gr.update(value=None, visible=False)

    _, test_idx_arr = split
    test_idx_arr = np.asarray(test_idx_arr)

    # Build export dataframe for ALL test (query) samples
    rows: List[Dict[str, Any]] = []
    for local_i, global_i in enumerate(test_idx_arr):
        if global_i >= len(features_df):
            continue
        row: Dict[str, Any] = {}
        # Add composition columns if available
        if comps_df is not None and global_i < len(comps_df):
            for c in comps_df.columns:
                row[c] = comps_df.iloc[global_i][c]
        # Add target if available
        if target is not None and global_i < len(target):
            row["target"] = target.iloc[global_i]
        # Add feature values
        for c in cols:
            if c in features_df.columns:
                row[c] = features_df.iloc[global_i][c]
        # Add OOD info
        if local_i < len(ood_res.composite_scores):
            row["OOD_Score"] = round(float(ood_res.composite_scores[local_i]), 4)
            row["is_OOD"] = bool(ood_res.is_ood[local_i])
        rows.append(row)

    if not rows:
        return gr.update(value=None, visible=False)

    # Columnar construction to avoid DataFrame fragmentation / SIGSEGV
    if rows:
        col_names = list(rows[0].keys())
        export_df = pd.DataFrame(
            {k: [r[k] for r in rows] for k in col_names}
        )
    else:
        export_df = pd.DataFrame()
    # Sort: OOD samples first, then by score descending
    export_df = export_df.sort_values(
        ["is_OOD", "OOD_Score"], ascending=[False, False],
    ).reset_index(drop=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = Path(tempfile.gettempdir()) / f"ood_{fs_key}_{timestamp}.csv"
    export_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")

    return gr.update(value=str(tmp_path), visible=True)


def _refresh_report_data(session: Dict[str, Any]) -> Tuple:
    report_path = session.get("report_path")
    if report_path is None or not Path(report_path).exists():
        return "*No report available.*", None
    content = Path(report_path).read_text(encoding="utf-8")
    return content, str(report_path)


# ---------------------------------------------------------------------------
# Literature search callback (module-level for testability)
# ---------------------------------------------------------------------------


def _do_literature_search(
    query: str, domain: str, task: str,
    inputs_scope: str, top: float,
    session: Dict[str, Any],
) -> Tuple:
    """Execute a literature search and return results for the GUI.

    Extracted to module level so it can be unit-tested independently
    of the Gradio app factory.
    """
    try:
        from hea_extrapolation_platform.literature_graph.search import (
            StructuredFilter,
        )
        from hea_extrapolation_platform.literature_graph.feature_recommender import (
            LiteratureFeatureRecommender,
        )

        # Fix #9: use cached engine
        engine = _get_literature_engine(session)

        sf = StructuredFilter(
            materials_domain=domain.strip() or None,
            task=task.strip() or None,
            inputs=inputs_scope.strip() or None,
        )

        results = engine.search(
            query, structured_filter=sf, top_n=int(top),
        )
        session["literature_results"] = results

        records = []
        for i, r in enumerate(results):
            wf = r.workflow
            records.append({
                "Rank": i + 1,
                "Paper ID": wf.paper_id,
                "Model": wf.model_name,
                "Family": wf.model_family,
                "Inputs": wf.inputs,
                "Split": wf.split_policy,
                "N": wf.data_size_n,
                "Key Features": ", ".join(
                    wf.key_features[:5],
                ),
                "Score": round(r.final_score, 4),
            })
        if records:
            col_names = list(records[0].keys())
            r_df = pd.DataFrame(
                {k: [rec[k] for rec in records] for k in col_names}
            )
        else:
            r_df = pd.DataFrame()

        _, feature_counts = engine.search_for_features(
            query, structured_filter=sf, top_n=int(top),
        )
        session["feature_counts"] = feature_counts
        freq_fig_val = plotly_feature_frequency(
            feature_counts,
        )

        recommender = LiteratureFeatureRecommender(engine)
        rec = recommender.recommend(
            query, structured_filter=sf,
        )
        session["feature_recommendation"] = rec

        # --- Build rich Feature Recommendation text ---
        rec_text = (
            "## Feature Recommendation (文献ベース特徴量推薦)\n\n"
            "文献データベースに登録されたワークフローの key_features を集計し、\n"
            "出現頻度の高い特徴量をベースセットに追加する推薦を生成します。\n\n"
            "### 推薦の仕組み\n"
            "1. 検索クエリに類似する文献ワークフローを上位5件取得\n"
            "2. 各ワークフローの key_features を出現頻度順に集計\n"
            "3. プラットフォームの FeatureCatalog に登録済みの特徴量のみを候補とする\n"
            "4. ベースセット(FS_BASE)に含まれない特徴量を最大5個追加\n\n"
        )
        rec_text += f"**推薦セット名**: `{rec.name}`\n\n"
        rec_text += (
            f"**ベース特徴量** ({len(rec.base_features)}個): "
            f"{', '.join(rec.base_features)}\n\n"
        )
        if rec.added_features:
            rec_text += (
                f"**文献から追加** ({len(rec.added_features)}個): "
                f"{', '.join(rec.added_features)}\n"
                f"  → これらは文献で頻出かつ FeatureCatalog に登録済みの特徴量です。\n\n"
            )
        else:
            rec_text += (
                "**文献から追加**: なし（ベースセットで文献の特徴量をカバー済み）\n\n"
            )
        if rec.unregistered_features:
            rec_text += (
                f"**未登録特徴量** (参考): "
                f"{', '.join(rec.unregistered_features)}\n"
                f"  → 文献で使用されているが本プラットフォーム未登録の特徴量。\n"
                f"    将来的に FeatureCatalog への追加を検討してください。\n\n"
            )
        if rec.feature_frequency:
            rec_text += "### 特徴量出現頻度 (上位10)\n"
            rec_text += "| 特徴量 | 出現回数 |\n|---|---|\n"
            sorted_freq = sorted(
                rec.feature_frequency.items(),
                key=lambda x: x[1], reverse=True,
            )[:10]
            for feat, count in sorted_freq:
                rec_text += f"| {feat} | {count} |\n"

        return r_df, freq_fig_val, rec_text, session

    except Exception:
        err = traceback.format_exc()
        return (
            pd.DataFrame(), None,
            f"Error:\n{err}", session,
        )


# ---------------------------------------------------------------------------
# Data Summary helpers
# ---------------------------------------------------------------------------

def _consolidate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a consolidated (single-block) copy of *df*.

    ``pd.DataFrame`` built from heterogeneous sources may carry one
    internal memory block per column (fragmented BlockManager).
    Operations like ``.describe()`` or ``.corr()`` on such a frame can
    trigger a SIGSEGV in the pandas/numpy C layer.

    Rebuilding from the underlying numpy array guarantees a single
    contiguous block for homogeneous-dtype frames (all float64).
    For mixed-dtype frames the columnar rebuild is used instead.
    """
    if df.empty:
        return df
    try:
        # Fast path: all-numeric frame -> single numpy block
        arr = df.to_numpy(dtype="float64", na_value=np.nan)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    except (ValueError, TypeError):
        # Mixed dtypes — rebuild from dict-of-lists (columnar)
        return pd.DataFrame(
            {c: df[c].tolist() for c in df.columns},
            index=df.index,
        )


def _refresh_data_summary(
    session: Dict[str, Any],
) -> Tuple:
    """Refresh the Data Summary tab from session state."""
    comps_df = session.get("compositions_df")
    features_df = session.get("features_df")
    target = session.get("target")

    # Consolidate features_df to a single memory block before heavy
    # numpy operations (.describe(), .corr()) that can SIGSEGV on
    # fragmented DataFrames.
    if features_df is not None and not features_df.empty:
        features_df = _consolidate_df(features_df)
        session["features_df"] = features_df

    stats_md = build_summary_stats_md(comps_df, features_df, target)

    if comps_df is not None and target is not None:
        target_fig = plotly_target_histogram(target)
        comp_fig = plotly_composition_heatmap(comps_df)
    else:
        target_fig = None
        comp_fig = None

    if features_df is not None and not features_df.empty:
        corr_fig = plotly_feature_correlation(features_df, target=target)
    else:
        corr_fig = None

    desc_df = pd.DataFrame()
    if features_df is not None and not features_df.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            desc_df = features_df.describe().round(3).reset_index()
        desc_df.rename(columns={"index": "Statistic"}, inplace=True)

    return stats_md, target_fig, comp_fig, corr_fig, desc_df


def _make_error_banner(title: str, detail: str) -> str:
    """Return a prominent Markdown/HTML error banner for the GUI."""
    return (
        f'<div style="background:#fff0f0; border:2px solid #e53e3e; '
        f'border-radius:8px; padding:16px; margin:8px 0;">'
        f'<span style="font-size:1.3em; font-weight:bold; color:#c53030;">'
        f'\u26a0\ufe0f {title}</span><br/>'
        f'<span style="color:#742a2a;">{detail}</span></div>'
    )


def _detect_element_columns(
    columns: List[str],
    available: set,
) -> Tuple[List[str], Dict[str, str]]:
    """Detect element columns with flexible name matching.

    Supports:
      - Exact element symbols: ``Fe``, ``Ni``, ``Co``
      - ``_frac`` suffix: ``Al_frac``, ``Fe_frac``
      - ``_at`` / ``_at%`` suffix: ``Al_at``, ``Fe_at%``
      - ``_wt`` / ``_wt%`` suffix: ``Al_wt``, ``Fe_wt%``
      - Case-insensitive matching: ``al``, ``FE``, ``ni_Frac``

    Returns
    -------
    elem_cols : list[str]
        Original column names that matched.
    col_to_elem : dict[str, str]
        Mapping from original column name to canonical element symbol.
    """
    import re
    # Common suffixes to strip (order matters — longest first)
    _SUFFIXES = re.compile(
        r'[_\s]+(frac|at%|at|wt%|wt|fraction|pct|percent|ratio)$',
        re.IGNORECASE,
    )
    # Map lowercased element symbols for case-insensitive lookup
    available_lower = {e.lower(): e for e in available}

    elem_cols: List[str] = []
    col_to_elem: Dict[str, str] = {}
    seen_elements: set = set()  # track already-matched canonical symbols

    for col in columns:
        # 1) Try exact match first
        if col in available:
            if col not in seen_elements:
                elem_cols.append(col)
                col_to_elem[col] = col
                seen_elements.add(col)
            continue

        # 2) Strip known suffixes and try again
        stripped = _SUFFIXES.sub('', col).strip()

        # 3) Case-insensitive lookup
        canon = available_lower.get(stripped.lower())
        if canon is not None and canon not in seen_elements:
            elem_cols.append(col)
            col_to_elem[col] = canon
            seen_elements.add(canon)

    return elem_cols, col_to_elem


def _handle_csv_upload(
    file_obj: Any,
    target_col: str,
    session: Dict[str, Any],
) -> Tuple:
    """Handle CSV file upload and compute features.

    Expected CSV format: element columns (atomic fractions summing to ~1)
    plus an optional target column.  Non-element columns that are not
    the target column are silently ignored during feature computation.

    Element column detection is flexible — supports bare symbols
    (``Fe``), ``_frac`` suffix (``Al_frac``), ``_at%``, ``_wt%``,
    and case-insensitive variants.

    Returns the same tuple shape as ``_refresh_data_summary`` plus the
    updated session.
    """
    if file_obj is None:
        return (
            build_summary_stats_md(None, None, None),
            None, None, None, pd.DataFrame(), session,
        )

    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        raw = pd.read_csv(file_path)

        if raw.empty:
            return (
                _make_error_banner(
                    "CSV is empty",
                    "アップロードされたCSVにデータ行がありません。",
                ),
                None, None, None, pd.DataFrame(), session,
            )

        from hea_extrapolation_platform.features import (
            _ElementDB,
            compute_features,
            FeatureSetName,
        )

        available = set(_ElementDB.available_elements())
        target_col_clean = target_col.strip()

        # Flexible element column detection
        elem_cols, col_to_elem = _detect_element_columns(
            list(raw.columns), available,
        )

        if not elem_cols:
            sample_cols = ", ".join(list(raw.columns)[:10])
            avail_str = ", ".join(sorted(available)[:15])
            return (
                _make_error_banner(
                    "元素列が見つかりません",
                    "CSVの列名から元素記号を検出できませんでした。<br/>"
                    f"<b>CSVの列名（先頭10列）</b>: <code>{sample_cols}</code><br/>"
                    f"<b>対応している元素記号</b>: <code>{avail_str}</code><br/><br/>"
                    "以下の命名規則に対応しています:<br/>"
                    "<code>Fe</code>, <code>Fe_frac</code>, "
                    "<code>Fe_at%</code>, <code>Fe_wt%</code> "
                    "（大文字小文字不問）",
                ),
                None, None, None, pd.DataFrame(), session,
            )

        # Build compositions list using canonical element symbols.
        # Track valid row indices to keep all DataFrames aligned
        # (rows with all-zero elements are dropped).
        valid_indices: List[int] = []
        compositions = []
        for idx, row in raw[elem_cols].iterrows():
            comp = {
                col_to_elem[c]: float(v)
                for c, v in row.items()
                if float(v) > 0
            }
            if comp:
                compositions.append(comp)
                valid_indices.append(idx)

        if not compositions:
            return (
                _make_error_banner(
                    "有効な組成データなし",
                    "検出された元素列の値がすべて 0 です。"
                    "原子分率が正の値を持つ行が必要です。",
                ),
                None, None, None, pd.DataFrame(), session,
            )

        # Composition DataFrame — rename to canonical element symbols
        comps_df = raw.loc[valid_indices, elem_cols].copy().reset_index(
            drop=True,
        )
        comps_df.columns = [col_to_elem[c] for c in comps_df.columns]

        # Compute features
        features_df = compute_features(compositions, feature_set=None)

        # Extract or synthesize target — aligned to valid rows
        if target_col_clean and target_col_clean in raw.columns:
            target = (
                raw.loc[valid_indices, target_col_clean]
                .copy()
                .reset_index(drop=True)
            )
            target.name = target_col_clean
        else:
            target = pd.Series(
                np.zeros(len(compositions)),
                name="(no target column)",
            )

        session["compositions_df"] = comps_df
        session["features_df"] = features_df
        session["target"] = target

        stats_md = build_summary_stats_md(comps_df, features_df, target)
        target_fig = plotly_target_histogram(target)
        comp_fig = plotly_composition_heatmap(comps_df)
        corr_fig = plotly_feature_correlation(features_df, target=target)
        # features_df is now built columnar (non-fragmented) in
        # compute_features(); no extra .copy() needed.
        desc_df = features_df.describe().round(3).reset_index()
        desc_df.rename(columns={"index": "Statistic"}, inplace=True)

        return stats_md, target_fig, comp_fig, corr_fig, desc_df, session

    except Exception:
        err = traceback.format_exc()
        return (
            _make_error_banner(
                "CSV読み込みエラー",
                f"<pre style='background:#fff5f5; padding:8px; "
                f"overflow-x:auto;'>{err}</pre>",
            ),
            None, None, None, pd.DataFrame(), session,
        )


# ---------------------------------------------------------------------------
# Progress bar HTML builder
# ---------------------------------------------------------------------------

def _build_log_html(log_text: str) -> str:
    """Wrap log text in a scrollable HTML container that auto-scrolls.

    Uses a unique element ID and inline ``<script>`` to scroll the
    container to the bottom each time Gradio updates the HTML.
    """
    # Escape HTML entities in log text, preserve newlines
    safe = html_mod.escape(log_text)
    lines_html = safe.replace("\n", "<br/>")
    return (
        '<div id="progress-log-container" style="'
        "background: #1e1e1e; color: #d4d4d4; "
        "font-family: 'Consolas','Monaco','Courier New',monospace; "
        "font-size: 12px; line-height: 1.5; "
        "padding: 10px 12px; border-radius: 6px; "
        "height: 320px; overflow-y: auto; "
        "white-space: pre-wrap; word-break: break-all; "
        "border: 1px solid #333;"
        '">'
        f"{lines_html}"
        "</div>"
        "<script>"
        "(function(){var c=document.getElementById('progress-log-container');"
        "if(c)c.scrollTop=c.scrollHeight;})();"
        "</script>"
    )


def _build_progress_bar_html(
    pct: int,
    label: str = "",
) -> str:
    """Return an HTML string for a styled progress bar.

    Parameters
    ----------
    pct:
        Percentage complete (0-100).
    label:
        Short text displayed below the bar (e.g. "Run 3/10 — FS_ALL").
    """
    pct = max(0, min(100, pct))
    # Pick colour based on progress
    if pct < 30:
        bar_color = "#4C72B0"      # blue
    elif pct < 70:
        bar_color = "#55A868"      # green
    else:
        bar_color = "#C44E52"      # red-ish accent for final stretch

    inner_text = f"{pct}%" if pct > 8 else ""
    return (
        f'<div style="margin: 10px 0;">'
        f'<div style="background: #e0e0e0; border-radius: 8px; '
        f'height: 28px; overflow: hidden;">'
        f'<div style="background: {bar_color}; '
        f'height: 100%; width: {pct}%; border-radius: 8px; '
        f'transition: width 0.3s; display: flex; '
        f'align-items: center; justify-content: center; '
        f'color: white; font-size: 13px; font-weight: bold;">'
        f'{inner_text}'
        f'</div></div>'
        f'<div style="text-align: center; margin-top: 4px; '
        f'color: #666; font-size: 12px;">{label}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Feature Selection helpers
# ---------------------------------------------------------------------------

def _build_fs_physical_origins_md() -> str:
    """Build a Markdown description of each feature set's physical origin."""
    from hea_extrapolation_platform.feature_selection import FS_PHYSICAL_ORIGINS

    lines = [
        "### 各特徴量セットの物理的起源\n",
        "各特徴量セットが捉える物理的メカニズムと、"
        "性能差が生じる理由を解説します。\n",
    ]
    for fs_name, description in FS_PHYSICAL_ORIGINS.items():
        lines.append(f"**{fs_name}**\n")
        lines.append(f"> {description}\n")
    lines.append(
        "\n---\n"
        "**性能差の物理的解釈**: "
        "FS_THERMO が FS_BASE より高スコアを示す場合、"
        "熱力学的安定性指標（H_mix, Omega）が目的変数と強い相関を持つことを意味します。"
        "逆に FS_ELECTRON が低スコアの場合、電子構造プロキシが "
        "対象系の支配的な強化機構を十分に捉えられていない可能性があります。\n\n"
        "**推奨**: まず FS_ALL で全特徴量を投入し XGBoost で非線形交互作用を捕捉した後、"
        "Lasso/ARD で有効特徴量を絞り込むアプローチが効果的です。"
    )
    return "\n".join(lines)


def _run_feature_selection_for_fs(
    fs_name: str,
    use_lasso: bool,
    use_aic: bool,
    use_bic: bool,
    use_ard: bool,
    session: Dict[str, Any],
) -> Tuple:
    """Run feature selection on the specified feature set.

    Returns (result_md, importance_plot, consensus_md, session).
    """
    import plotly.graph_objects as go

    features_df = session.get("features_df")
    target = session.get("target")

    if features_df is None or target is None:
        return (
            "*先に Config & Run で解析を実行してください。*",
            None,
            "*No data available.*",
            session,
        )

    # Determine which columns belong to this feature set
    from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
    try:
        fs_enum = FeatureSetName(fs_name)
    except ValueError:
        return (
            f"*Unknown feature set: {fs_name}*",
            None, "*Error*", session,
        )

    fs_cols = FeatureCatalog.columns(fs_enum)
    available_cols = [c for c in fs_cols if c in features_df.columns]
    if not available_cols:
        return (
            f"*Feature set {fs_name} has no matching columns in current data.*",
            None, "*Error*", session,
        )

    X = features_df[available_cols].copy()
    y = target.copy()

    # Build method list
    methods: List[str] = []
    if use_lasso:
        methods.append("Lasso")
    if use_aic:
        methods.append("AIC")
    if use_bic:
        methods.append("BIC")
    if use_ard:
        methods.append("ARD")

    if not methods:
        return (
            "*少なくとも1つの特徴量選択手法を選択してください。*",
            None, "*No methods selected.*", session,
        )

    from hea_extrapolation_platform.feature_selection import run_feature_selection
    summary = run_feature_selection(
        X, y, methods=methods, consensus_threshold=2, feature_set=fs_name,
    )

    # Store in session
    fs_results = session.get("feature_selection_results", {})
    fs_results[fs_name] = summary
    session["feature_selection_results"] = fs_results

    # --- Build result markdown ---
    md_lines = [f"### {fs_name} 特徴量選択結果\n"]
    md_lines.append(f"入力特徴量数: **{X.shape[1]}** | 手法数: **{len(methods)}**\n")
    md_lines.append("| 手法 | 選択数 | 選択された特徴量 |")
    md_lines.append("|---|---|---|")
    for method_name, result in summary.results.items():
        feat_str = ", ".join(result.selected_features[:10])
        if len(result.selected_features) > 10:
            feat_str += f" ... (+{len(result.selected_features) - 10})"
        md_lines.append(
            f"| {method_name} | {result.n_selected} | {feat_str} |"
        )
    result_md = "\n".join(md_lines)

    # --- Build importance bar chart ---
    # Aggregate scores across methods for a combined view
    all_features = list(X.columns)
    combined_scores: Dict[str, float] = {f: 0.0 for f in all_features}
    for result in summary.results.values():
        max_score = max(result.all_scores.values()) if result.all_scores else 1.0
        if max_score <= 0:
            max_score = 1.0
        for feat, score in result.all_scores.items():
            if feat in combined_scores:
                combined_scores[feat] += score / max_score

    # Sort by combined score, show top 20
    sorted_feats = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True,
    )[:20]
    feat_names = [f[0] for f in sorted_feats]
    feat_scores = [f[1] for f in sorted_feats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feat_scores,
        y=feat_names,
        orientation="h",
        marker_color="#4C72B0",
    ))
    fig.update_layout(
        title=f"Feature Importance ({fs_name}) - Combined across {len(methods)} methods",
        xaxis_title="Normalised Importance (sum across methods)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(feat_names) * 25),
        margin=dict(l=150),
    )

    # --- Build consensus markdown ---
    consensus_lines = ["### Consensus Features (コンセンサス特徴量)\n"]
    if summary.consensus_features:
        consensus_lines.append(
            f"**{len(summary.consensus_features)}個** の特徴量が "
            f"{summary.consensus_threshold}手法以上で選択されました:\n"
        )
        for feat in summary.consensus_features:
            consensus_lines.append(f"- `{feat}`")
        consensus_lines.append(
            "\n> コンセンサス特徴量は複数の異なるアルゴリズムで"
            "一貫して選択されたため、頑健な予測因子である可能性が高いです。"
        )
    else:
        consensus_lines.append(
            f"*{summary.consensus_threshold}手法以上で共通して選択された"
            "特徴量はありませんでした。閾値を下げるか、手法を追加してください。*"
        )

    # Add physical origin context
    from hea_extrapolation_platform.feature_selection import FS_PHYSICAL_ORIGINS
    origin = FS_PHYSICAL_ORIGINS.get(fs_name, "")
    if origin:
        consensus_lines.append(f"\n---\n**{fs_name} の物理的背景**:\n> {origin}")

    consensus_md = "\n".join(consensus_lines)

    return (result_md, fig, consensus_md, session)


# ---------------------------------------------------------------------------
# Main App Factory
# ---------------------------------------------------------------------------

# Current PR / build version tag shown in the GUI title bar.
_GUI_VERSION_TAG = "PR#114"


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    # Fix #5: theme passed to gr.Blocks() constructor
    with gr.Blocks(
        title=f"Extrapolation Discovery Platform ({_GUI_VERSION_TAG})",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            f"# Extrapolation Discovery Platform &ensp;"
            f"<small style='color:#888;'>({_GUI_VERSION_TAG})</small>\n"
            "Feature Validity Evaluation & OOD Detection Dashboard\n\n"
            "**使い方**: Config & Run タブでパラメータを設定し、"
            "\"Run Analysis\" を押すと全タブが自動更新されます。"
        )

        # Fix #1: per-user session state via gr.State
        state = gr.State(_empty_session)

        with gr.Tabs():
            # --- Tab 1: Config & Run ---
            with gr.Tab("Config & Run"):
                gr.Markdown(
                    "## Analysis Configuration & Execution\n\n"
                    "このプラットフォームは、複数のMLワークフロー×特徴量セット×"
                    "分割ポリシーの組み合わせを網羅的に実行し、"
                    "特徴量の妥当性を評価します。\n"
                    "OOD検出で外挿危険領域を特定し、文献検索で最適記述子を探索します。"
                )

                # --- Data Source Selection ---
                gr.Markdown(
                    "### データソース選択\n"
                    "**方法 A**: CSVファイルをアップロードして独自データで解析\n\n"
                    "**方法 B**: CSVを指定しなければ、組込みサンプルデータを自動生成"
                )
                with gr.Row():
                    run_csv_upload = gr.File(
                        label="\U0001F4C2 CSV Upload (任意)",
                        file_types=[".csv"],
                    )
                    run_csv_target = gr.Textbox(
                        label="Target Column Name",
                        value="yield_strength_MPa",
                        info="CSVの目的変数の列名",
                    )
                csv_status_html = gr.HTML(
                    value=(
                        '<div style="padding:8px 12px; background:#e8f4fd; '
                        'border-left:4px solid #2196F3; border-radius:4px; '
                        'margin:4px 0; font-size:14px;">'
                        '\U0001F4CA <b>データソース</b>: '
                        'サンプルデータ自動生成モード（CSVをアップロードすると切替）'
                        '</div>'
                    ),
                )

                with gr.Row():
                    with gr.Column():
                        seeds_input = gr.Textbox(
                            label="Seeds (space-separated)",
                            value="42 123 456",
                            info="Random seeds for reproducibility",
                        )
                        n_samples = gr.Slider(
                            minimum=50, maximum=1000, value=200, step=50,
                            label="Number of Samples",
                            info="CSVアップロード時は無視されます",
                        )
                        quick_mode = gr.Checkbox(
                            label="Quick Mode (reduced HPO)",
                            value=True,
                            info="Use reduced hyperparameter grids",
                        )
                    with gr.Column():
                        exclude_elements = gr.Textbox(
                            label="Exclude Elements (space-separated)",
                            value="Co Ni Ti",
                            info="Elements for ElementExclusion splits",
                        )
                        skip_literature = gr.Checkbox(
                            label="Skip Literature Search",
                            value=False,
                        )
                        skip_plots = gr.Checkbox(
                            label="Skip Static Plots (matplotlib)",
                            value=True,
                            info="Skip PNG generation (Plotly always available)",
                        )

                # Update CSV status when file is uploaded/removed
                def _update_csv_status(file_obj: Any) -> str:
                    if file_obj is None:
                        return (
                            '<div style="padding:8px 12px; background:#e8f4fd; '
                            'border-left:4px solid #2196F3; border-radius:4px; '
                            'margin:4px 0; font-size:14px;">'
                            '\U0001F4CA <b>データソース</b>: '
                            'サンプルデータ自動生成モード'
                            '（CSVをアップロードすると切替）'
                            '</div>'
                        )
                    fname = (
                        file_obj.name.split("/")[-1]
                        if hasattr(file_obj, "name")
                        else "uploaded.csv"
                    )
                    return (
                        '<div style="padding:8px 12px; background:#e8f5e9; '
                        'border-left:4px solid #4CAF50; border-radius:4px; '
                        'margin:4px 0; font-size:14px;">'
                        f'\u2705 <b>データソース</b>: '
                        f'<code>{html_mod.escape(fname)}</code> を使用します'
                        '（Number of Samples は無視されます）'
                        '</div>'
                    )

                run_csv_upload.change(
                    fn=_update_csv_status,
                    inputs=[run_csv_upload],
                    outputs=[csv_status_html],
                )

                # --- ML Algorithm Selection ---
                with gr.Accordion(
                    "ML Algorithm Selection (ワークフロー選択)",
                    open=True,
                ):
                    gr.Markdown(
                        "実行するMLワークフローを選択してください。\n\n"
                        "- **WF-LIN** (Ridge回帰): 特徴量の符号検証・リーク検出に有効\n"
                        "- **WF-XGB** (XGBoost + GridSearchCV): 非線形交互作用を捕捉\n"
                        "- **WF-ENS** (Seed-varied Ensemble): 予測不確実性の定量化"
                    )
                    with gr.Row():
                        wf_lin_check = gr.Checkbox(
                            label="WF-LIN (Ridge Regression)",
                            value=True,
                        )
                        wf_xgb_check = gr.Checkbox(
                            label="WF-XGB (XGBoost + HPO)",
                            value=True,
                        )
                        wf_ens_check = gr.Checkbox(
                            label="WF-ENS (Ensemble UQ)",
                            value=True,
                        )

                # --- Feature Selection Method Selection ---
                with gr.Accordion(
                    "Feature Selection Methods (特徴量選択アルゴリズム)",
                    open=False,
                ):
                    gr.Markdown(
                        "解析完了後に FS Comparison タブで実行できる"
                        "特徴量選択アルゴリズムを選択してください。\n\n"
                        "- **Lasso** (L1正則化): 不要な特徴量の係数をゼロに押す\n"
                        "- **AIC** (赤池情報量規準): 前進ステップワイズ選択。複雑さへのペナルティが小さい\n"
                        "- **BIC** (ベイズ情報量規準): AICより強いペナルティで簡素なモデルを選好\n"
                        "- **ARD** (自動関連度決定): ベイズ的スパース回帰で特徴量を自動剣定"
                    )
                    with gr.Row():
                        fs_lasso_check = gr.Checkbox(
                            label="Lasso (L1)", value=True,
                        )
                        fs_aic_check = gr.Checkbox(
                            label="AIC (Forward)", value=True,
                        )
                        fs_bic_check = gr.Checkbox(
                            label="BIC (Forward)", value=True,
                        )
                        fs_ard_check = gr.Checkbox(
                            label="ARD (Bayesian)", value=True,
                        )

                # Integrations are always enabled behind the scenes.
                gr.Markdown(
                    "All tracking (MLflow), feature management "
                    "(Feast), and workflow execution (MInt) are "
                    "**automatically enabled**.  "
                    "Results are recorded and managed transparently."
                )

                run_btn = gr.Button(
                    "▶ Run Analysis (解析実行)", variant="primary", size="lg",
                )

                # --- Progress bar (HTML-based for real-time updates) ---
                progress_bar_html = gr.HTML(
                    value=(
                        '<div style="margin: 10px 0;">'
                        '<div style="background: #e0e0e0; border-radius: 8px; '
                        'height: 28px; overflow: hidden;">'
                        '<div id="exp-progress" style="background: #4C72B0; '
                        'height: 100%; width: 0%; border-radius: 8px; '
                        'transition: width 0.3s; display: flex; '
                        'align-items: center; justify-content: center; '
                        'color: white; font-size: 13px; font-weight: bold;">'
                        '</div></div>'
                        '<div style="text-align: center; margin-top: 4px; '
                        'color: #666; font-size: 12px;">Waiting to start...</div>'
                        '</div>'
                    ),
                    label="Progress",
                )

                progress_log = gr.HTML(
                    value=_build_log_html("Waiting to start..."),
                    label="Progress Log",
                )

            # --- Tab 2: Data Summary (Statistics only) ---
            with gr.Tab("Data Summary"):
                gr.Markdown(
                    "## Data Summary\n"
                    "データセットの概要統計を確認できます。"
                    "CSVアップロードは **Config & Run** タブから行ってください。"
                )

                summary_stats_md = gr.Markdown(
                    build_summary_stats_md(None, None, None),
                )

                with gr.Row():
                    target_hist_plot = gr.Plot(
                        label="Target Distribution",
                    )
                    comp_bar_plot = gr.Plot(
                        label="Element Composition (Mean +/- Std)",
                    )

                corr_heatmap_plot = gr.Plot(
                    label="Feature Correlation Matrix",
                )

                with gr.Accordion(
                    "Full Feature Statistics", open=False,
                ):
                    feature_stats_table = gr.Dataframe(
                        label="Descriptive Statistics (all features)",
                    )

                summary_refresh_btn = gr.Button(
                    "Refresh Summary", variant="primary",
                )

                summary_outputs = [
                    summary_stats_md, target_hist_plot,
                    comp_bar_plot, corr_heatmap_plot,
                    feature_stats_table,
                ]
                summary_refresh_btn.click(
                    fn=_refresh_data_summary,
                    inputs=[state],
                    outputs=summary_outputs,
                )


            # --- Tab 3: Dashboard ---
            with gr.Tab("Dashboard"):
                gr.Markdown("## Dashboard -- KPIs & Feature Validity")
                gr.Markdown(
                    "**Config & Run** タブで解析を実行すると、"
                    "このページに結果が表示されます。"
                )

                with gr.Row():
                    kpi_runs = gr.Textbox(
                        label="Total Runs", value="0", interactive=False,
                    )
                    kpi_best_fs = gr.Textbox(
                        label="Best Feature Set", value="--", interactive=False,
                    )
                    kpi_best_score = gr.Textbox(
                        label="Best Total Score", value="--", interactive=False,
                    )
                    kpi_ood_count = gr.Textbox(
                        label="OOD Samples", value="--", interactive=False,
                    )

                # --- Integration Status (collapsible, hidden by default) ---
                with gr.Accordion(
                    "System Details (MLflow / Feast / MInt)",
                    open=False,
                ):
                    integration_status = gr.Markdown(
                        _build_integration_status_md(None),
                    )

                validity_plot = gr.Plot(label="Feature Validity Ranking")
                heatmap_plot = gr.Plot(label="Performance Heatmap (RMSE Test)")
                heatmap_metric = gr.Dropdown(
                    choices=[
                        "rmse_test", "rmse_train", "mae_test",
                        "mae_train", "r2_test", "r2_train",
                    ],
                    value="rmse_test",
                    label="Heatmap Metric",
                )

                dash_refresh_btn = gr.Button(
                    "Refresh Dashboard", variant="primary",
                )
                dash_outputs = [
                    kpi_runs, kpi_best_fs, kpi_best_score,
                    kpi_ood_count, integration_status,
                    validity_plot, heatmap_plot,
                ]
                dash_refresh_btn.click(
                    fn=_refresh_dashboard_data,
                    inputs=[heatmap_metric, state],
                    outputs=dash_outputs,
                )
                heatmap_metric.change(
                    fn=_refresh_dashboard_data,
                    inputs=[heatmap_metric, state],
                    outputs=dash_outputs,
                )

            # --- Tab 4: Results & FS Comparison ---
            with gr.Tab("Results"):
                gr.Markdown(
                    "## Analysis Results & Physical Interpretation\n\n"
                    "解析結果の数値データ・パリティプロット・FS比較・"
                    "物理的考察を統合表示します。"
                )

                # --- Physical Interpretation + FS Comparison ---
                with gr.Accordion(
                    "物理的考察 & FS 比較サマリー "
                    "(Physical Interpretation & FS Comparison)",
                    open=True,
                ):
                    results_interp_md = gr.Markdown(
                        "*解析を実行すると、ここに物理的考察と "
                        "FS比較サマリーが表示されます。*"
                    )
                    results_fs_bar_plot = gr.Plot(
                        label="Feature Set 性能比較 (Grouped Bar)",
                    )

                # --- Filters ---
                with gr.Row():
                    filter_wf = gr.Dropdown(
                        choices=["All"], value="All", label="Workflow Filter",
                    )
                    filter_fs = gr.Dropdown(
                        choices=["All"], value="All", label="Feature Set Filter",
                    )
                    filter_sp = gr.Dropdown(
                        choices=["All"], value="All", label="Split Policy Filter",
                    )

                validity_table = gr.Dataframe(label="Feature Validity Ranking")
                results_table = gr.Dataframe(label="Run Results")
                parity_plot = gr.Plot(label="Parity Plot")

                res_refresh_btn = gr.Button(
                    "Refresh Results", variant="primary",
                )
                res_outputs = [
                    filter_wf, filter_fs, filter_sp,
                    validity_table, results_table, parity_plot,
                    results_interp_md, results_fs_bar_plot,
                ]
                res_refresh_btn.click(
                    fn=_refresh_results_data,
                    inputs=[filter_wf, filter_fs, filter_sp, state],
                    outputs=res_outputs,
                )
                for dropdown in [filter_wf, filter_fs, filter_sp]:
                    dropdown.change(
                        fn=_refresh_results_data,
                        inputs=[filter_wf, filter_fs, filter_sp, state],
                        outputs=res_outputs,
                    )

            # --- Tab 5: OOD Map (Out-of-Distribution) ---
            with gr.Tab("OOD Map"):
                gr.Markdown(
                    "## OOD (Out-of-Distribution) Map & Candidates",
                )

                fs_selector = gr.Dropdown(
                    choices=[
                        "FS_BASE", "FS_THERMO", "FS_SIZE",
                        "FS_ELECTRON", "FS_ALL", "FS_MAGPIE",
                    ],
                    value="FS_ALL",
                    label="Feature Set for OOD Map",
                )
                ood_plot = gr.Plot(label="OOD Map (PCA)")
                with gr.Row():
                    ood_summary = gr.Textbox(
                        label="OOD Summary", interactive=False,
                    )
                ood_candidates = gr.Dataframe(label="Top OOD Candidates")

                with gr.Row():
                    ood_refresh_btn = gr.Button(
                        "Refresh OOD Map", variant="primary",
                    )
                    ood_csv_btn = gr.Button(
                        "\u2b07 Download OOD Candidates CSV",
                        variant="secondary",
                    )
                ood_csv_file = gr.File(
                    label="OOD CSV Download", visible=False,
                )

                ood_outputs = [ood_plot, ood_summary, ood_candidates]
                ood_refresh_btn.click(
                    fn=_refresh_ood_data,
                    inputs=[fs_selector, state],
                    outputs=ood_outputs,
                )
                fs_selector.change(
                    fn=_refresh_ood_data,
                    inputs=[fs_selector, state],
                    outputs=ood_outputs,
                )
                ood_csv_btn.click(
                    fn=_export_ood_csv,
                    inputs=[fs_selector, state],
                    outputs=[ood_csv_file],
                )

            # --- Tab 6: FS Comparison (Feature Selection + Physical Origins) ---
            with gr.Tab("FS Comparison"):
                gr.Markdown(
                    "## Feature Set Comparison & Feature Selection\n\n"
                    "各特徴量セットの物理的起源と、特徴量選択アルゴリズムの結果を表示します。"
                )

                # --- Physical Origin Descriptions ---
                with gr.Accordion(
                    "特徴量セットの物理的起源 (Physical Origins)",
                    open=True,
                ):
                    fs_origin_md = gr.Markdown(
                        _build_fs_physical_origins_md(),
                    )

                # --- Feature Selection ---
                with gr.Accordion(
                    "特徴量選択結果 (Feature Selection Results)",
                    open=True,
                ):
                    fs_comparison_selector = gr.Dropdown(
                        choices=[
                            "FS_BASE", "FS_THERMO", "FS_SIZE",
                            "FS_ELECTRON", "FS_ALL", "FS_MAGPIE",
                        ],
                        value="FS_ALL",
                        label="特徴量セット選択",
                    )
                    run_fs_btn = gr.Button(
                        "▶ Run Feature Selection (特徴量選択実行)",
                        variant="primary",
                    )
                    fs_result_md = gr.Markdown(
                        "*まだ特徴量選択を実行していません。"
                        "先に Config & Run で解析を実行してください。*"
                    )
                    fs_importance_plot = gr.Plot(
                        label="Feature Importance (selected methods)",
                    )
                    fs_consensus_md = gr.Markdown(
                        "*Consensus features will appear here.*"
                    )

                    run_fs_btn.click(
                        fn=_run_feature_selection_for_fs,
                        inputs=[
                            fs_comparison_selector,
                            fs_lasso_check, fs_aic_check,
                            fs_bic_check, fs_ard_check,
                            state,
                        ],
                        outputs=[
                            fs_result_md, fs_importance_plot,
                            fs_consensus_md, state,
                        ],
                    )

            # --- Tab 7: Literature Search ---
            with gr.Tab("Literature Search"):
                gr.Markdown(
                    "## Literature Search -- Embedding + Structured Filters",
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Search Query",
                            value="composition only yield strength HEA",
                            lines=2,
                            info="Natural language or workflow-text query",
                        )
                    with gr.Column(scale=1):
                        domain_filter = gr.Textbox(
                            label="Domain", value="HEA",
                        )
                        task_filter = gr.Textbox(
                            label="Task", value="yield_strength",
                        )

                with gr.Row():
                    # Fix #4: choices match InputScope enum values
                    inputs_filter = gr.Dropdown(
                        choices=[
                            "",
                            "composition_only",
                            "composition+process",
                            "composition+calphad",
                            "composition+microstructure",
                            "full",
                        ],
                        value="",
                        label="Inputs Scope",
                    )
                    lit_top_n = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Top N",
                    )

                search_btn = gr.Button(
                    "Search Literature", variant="primary",
                )
                lit_results_table = gr.Dataframe(label="Search Results")
                freq_plot = gr.Plot(label="Feature Frequency")
                lit_recommendation = gr.Markdown(
                    value="*Search to see feature recommendations.*",
                )

                search_btn.click(
                    fn=_do_literature_search,
                    inputs=[
                        query_input, domain_filter, task_filter,
                        inputs_filter, lit_top_n, state,
                    ],
                    outputs=[
                        lit_results_table, freq_plot,
                        lit_recommendation, state,
                    ],
                )

            # --- Tab 8: Report ---
            with gr.Tab("Report"):
                gr.Markdown(
                    "## Analysis Report -- Markdown Preview & Download",
                )
                report_md = gr.Markdown(
                    value="*まだレポートが生成されていません。"
                    "Config & Run で解析を実行してください。*",
                )
                download_btn = gr.File(label="Download Report (.md)")

                rpt_refresh_btn = gr.Button(
                    "Refresh Report", variant="primary",
                )
                rpt_outputs = [report_md, download_btn]
                rpt_refresh_btn.click(
                    fn=_refresh_report_data,
                    inputs=[state],
                    outputs=rpt_outputs,
                )

        # ---------------------------------------------------------------
        # Run experiment generator  -- Fix #7 (streaming progress)
        # ---------------------------------------------------------------

        def run_experiment(
            seeds_str: str,
            n_samp: float,
            quick: bool,
            excl_str: str,
            skip_lit: bool,
            skip_plt: bool,
            use_wf_lin: bool,
            use_wf_xgb: bool,
            use_wf_ens: bool,
            csv_file: Any,
            csv_target: str,
            session: Dict[str, Any],
        ) -> Generator:
            """Generator that yields incremental progress + state.

            Using a generator enables Gradio to stream updates to the
            progress_log while the analysis runs (fix #7).
            The final yield refreshes all cross-tab components (#8)
            and the Data Summary tab.

            Progress bar phases (approximate):
              0%   Starting
             10%   Dataset generated
             20%   Runner initialised
             60%   Analysis completed
             70%   Registry exported
             80%   Static plots done
             90%   Literature search done
            100%   Report generated — complete
            """
            log_lines: List[str] = []

            def log(msg: str) -> None:
                log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

            # Reset session — intentionally discard the incoming gr.State
            # value and start fresh so that stale data from a previous run
            # does not leak into the new experiment.
            session = _empty_session()  # noqa: F841 (shadows parameter)

            # Placeholder outputs for cross-tab components
            empty_dash = ("0", "--", "--", "--",
                          _build_integration_status_md(None), None, None)
            empty_res = (
                gr.update(), gr.update(), gr.update(),
                pd.DataFrame(), pd.DataFrame(), None,
                "*Running...*", None,
            )
            empty_ood = (None, "", pd.DataFrame())
            empty_rpt = ("*Running...*", None)
            empty_summary = (
                build_summary_stats_md(None, None, None),
                None, None, None, pd.DataFrame(),
            )

            def _yield_progress(
                log_text: str,
                pct: int = 0,
                bar_label: str = "",
                summary_tuple: Optional[Tuple] = None,
            ) -> Tuple:
                nonlocal last_pct
                last_pct = pct
                bar_html = _build_progress_bar_html(pct, bar_label)
                log_html = _build_log_html(log_text)
                summ = summary_tuple if summary_tuple else empty_summary
                return (
                    bar_html, log_html, session,
                    *summ,
                    *empty_dash, *empty_res, *empty_ood, *empty_rpt,
                )

            succeeded = False
            last_pct = 0

            try:
                log("Starting analysis...")
                yield _yield_progress(
                    "\n".join(log_lines), 0, "Initialising...",
                )

                # Validate seeds input
                try:
                    seeds = [
                        int(s.strip())
                        for s in seeds_str.split()
                        if s.strip()
                    ]
                except ValueError:
                    log(
                        "Error: Seeds must be space-separated integers "
                        "(e.g. 42 123 456). Got: " + repr(seeds_str)
                    )
                    yield _yield_progress(
                        "\n".join(log_lines), 0, "Error",
                    )
                    return

                excl = [
                    e.strip()
                    for e in excl_str.split()
                    if e.strip()
                ]

                if not seeds:
                    log(
                        "Error: No valid seeds provided. "
                        "Enter space-separated integers (e.g. 42 123 456)."
                    )
                    yield _yield_progress(
                        "\n".join(log_lines), 0, "Error",
                    )
                    return

                # Fix #2: cast slider float -> int
                n_samp_int = int(n_samp)

                # 1. Dataset loading / generation  (0% -> 10%)
                if csv_file is not None:
                    # --- Use uploaded CSV ---
                    log("Loading uploaded CSV...")
                    yield _yield_progress(
                        "\n".join(log_lines), 5,
                        "Loading CSV...",
                    )
                    csv_result = _handle_csv_upload(
                        csv_file, csv_target, session,
                    )
                    # _handle_csv_upload returns
                    # (stats_md, fig, fig, fig, df, session)
                    # — session is updated in-place with
                    # compositions_df, features_df, target.
                    session = csv_result[-1]
                    comps_df = session.get("compositions_df")
                    features_df = session.get("features_df")
                    target = session.get("target")
                    if comps_df is None or features_df is None:
                        # CSV loading failed — the error is in
                        # csv_result[0] (Markdown banner)
                        log(
                            "ERROR: CSV loading failed. "
                        )
                        # Surface the error summary from
                        # _handle_csv_upload to the Data Summary tab
                        error_summary = csv_result[:5]
                        yield _yield_progress(
                            "\n".join(log_lines), 0, "CSV Error",
                            summary_tuple=error_summary,
                        )
                        return
                    log(
                        f"CSV loaded: {len(target)} samples, "
                        f"{features_df.shape[1]} features"
                    )
                else:
                    # --- Generate sample dataset ---
                    from hea_extrapolation_platform.dataset import (
                        generate_hea_dataset,
                    )
                    log(
                        f"Generating dataset: "
                        f"n={n_samp_int}, seed={seeds[0]}"
                    )
                    yield _yield_progress(
                        "\n".join(log_lines), 5,
                        f"Generating {n_samp_int} samples...",
                    )

                    comps_df, features_df, target = generate_hea_dataset(
                        n_samples=n_samp_int, seed=seeds[0],
                    )
                    session["compositions_df"] = comps_df
                    session["features_df"] = features_df
                    session["target"] = target
                    log(
                        f"Dataset: {len(target)} samples, "
                        f"{features_df.shape[1]} features"
                    )
                # Build data-summary tuple now that we have data
                data_summary = _refresh_data_summary(session)

                yield _yield_progress(
                    "\n".join(log_lines), 10,
                    "Dataset ready",
                    summary_tuple=data_summary,
                )

                # 2. Run experiments  (10% -> 60%)
                from hea_extrapolation_platform.runner import (
                    ExperimentRunner,
                )
                # Build selected workflow list
                selected_wfs: List[str] = []
                if use_wf_lin:
                    selected_wfs.append("WF-LIN")
                if use_wf_xgb:
                    selected_wfs.append("WF-XGB")
                if use_wf_ens:
                    selected_wfs.append("WF-ENS")
                if not selected_wfs:
                    selected_wfs = ["WF-LIN", "WF-XGB", "WF-ENS"]
                    log("No workflows selected; using all.")

                log(
                    f"Running analysis: seeds={seeds}, quick={quick}, "
                    f"workflows={selected_wfs}"
                )
                yield _yield_progress(
                    "\n".join(log_lines), 20,
                    f"Training {len(selected_wfs)} workflow(s)...",
                    summary_tuple=data_summary,
                )

                runner = ExperimentRunner(
                    seeds=seeds, quick=quick, exclude_elements=excl,
                    use_mlflow=True,
                    use_feast=True,
                    use_mint=True,
                )

                # --- Real-time log capture via threading ---
                # Install a custom logging handler on the package
                # logger so that all INFO+ messages from runner.py,
                # ood.py, etc. are captured into log_lines and
                # streamed to the GUI in real time.
                log_q: queue.Queue[str] = queue.Queue()
                _cur_pct = [20]  # mutable container for progress %

                class _LogCapture(logging.Handler):
                    """Handler that forwards log records to a queue."""
                    def emit(self, record: logging.LogRecord) -> None:
                        try:
                            log_q.put(self.format(record))
                        except Exception:
                            pass

                _cap = _LogCapture()
                _cap.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S",
                    ),
                )
                _cap.setLevel(logging.INFO)
                pkg_logger = logging.getLogger(
                    "hea_extrapolation_platform",
                )
                pkg_logger.addHandler(_cap)
                pkg_logger.setLevel(logging.INFO)

                def _progress_cb(
                    completed: int, total: int, message: str,
                ) -> None:
                    if total > 0:
                        _cur_pct[0] = 20 + int(40 * completed / total)

                # Run the experiment in a background thread so
                # the generator can yield log updates in real time.
                _result_holder: Dict[str, Any] = {}

                def _run_in_thread() -> None:
                    try:
                        r, s, o = runner.run(
                            comps_df, features_df, target,
                            progress_callback=_progress_cb,
                            selected_workflows=selected_wfs,
                        )
                        _result_holder["runs"] = r
                        _result_holder["scores"] = s
                        _result_holder["ood"] = o
                    except Exception:
                        _result_holder["error"] = traceback.format_exc()

                thread = threading.Thread(
                    target=_run_in_thread, daemon=True,
                )
                thread.start()

                # Poll for new log messages and yield them
                while thread.is_alive():
                    _drained = False
                    while not log_q.empty():
                        try:
                            msg = log_q.get_nowait()
                            log_lines.append(msg)
                            _drained = True
                        except queue.Empty:
                            break
                    if _drained:
                        yield _yield_progress(
                            "\n".join(log_lines),
                            _cur_pct[0],
                            f"Training... ({_cur_pct[0]}%)",
                            summary_tuple=data_summary,
                        )
                    time.sleep(0.5)

                # Drain remaining log messages after thread ends
                while not log_q.empty():
                    try:
                        log_lines.append(log_q.get_nowait())
                    except queue.Empty:
                        break

                # Remove the capture handler
                pkg_logger.removeHandler(_cap)

                # Check for errors from the thread
                if "error" in _result_holder:
                    log(f"ERROR:\n{_result_holder['error']}")
                    yield _yield_progress(
                        "\n".join(log_lines), _cur_pct[0],
                        "Error",
                        summary_tuple=data_summary,
                    )
                    return

                runs = _result_holder["runs"]
                scores = _result_holder["scores"]
                ood_results = _result_holder["ood"]

                # Free thread-local temporaries and consolidate memory
                # before the heavy post-processing phase that builds
                # DataFrames for every tab.
                del _result_holder
                gc.collect()

                session["runs"] = runs
                session["validity_scores"] = scores
                session["ood_results"] = ood_results
                session["ood_split_indices"] = runner.ood_split_indices
                session["runner"] = runner

                log(f"Completed: {len(runs)} runs")

                # Log integration status (transparent to user)
                tracked = runner.tracker.list_runs()
                log(f"Analysis tracking: {len(tracked)} run(s) recorded")
                fs_sets = runner.feature_store.list_feature_sets()
                log(f"Feature store: {len(fs_sets)} feature set(s) managed")
                if runner.mint_registry is not None:
                    n_mint = len(runner.mint_registry.list_workflows())
                    log(f"Workflow engine: {n_mint} workflow(s) executed")

                if scores:
                    log(
                        f"Best feature set: {scores[0].feature_set} "
                        f"(score={scores[0].total:.4f})"
                    )

                for fs_key, ood_res in ood_results.items():
                    log(
                        f"OOD [{fs_key}]: "
                        f"{ood_res.n_ood}/{ood_res.n_total} "
                        f"({ood_res.ood_ratio:.1%})"
                    )
                yield _yield_progress(
                    "\n".join(log_lines), 60,
                    f"{len(runs)} runs complete",
                    summary_tuple=data_summary,
                )

                # 3. Export registry  -- Fix #12: timestamp dir  (60% -> 70%)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("results") / ts
                out_dir.mkdir(parents=True, exist_ok=True)
                runner.export(out_dir)
                log(
                    f"Run registry exported to "
                    f"{out_dir / 'run_registry.json'}"
                )
                yield _yield_progress(
                    "\n".join(log_lines), 70,
                    "Results exported",
                    summary_tuple=data_summary,
                )

                # 4. Static plots (optional)  (70% -> 80%)
                figure_paths: Dict[str, Path] = {}
                if not skip_plt:
                    from hea_extrapolation_platform.visualization import (
                        plot_validity_ranking,
                        plot_performance_heatmap,
                        plot_parity,
                    )
                    fig_dir = out_dir / "figures"
                    figure_paths["Validity"] = plot_validity_ranking(
                        scores, fig_dir,
                    )
                    figure_paths["Heatmap"] = plot_performance_heatmap(
                        runs, fig_dir,
                    )
                    figure_paths["Parity"] = plot_parity(runs, fig_dir)
                    log("Static plots saved.")
                    yield _yield_progress(
                        "\n".join(log_lines), 80,
                        "Plots saved",
                        summary_tuple=data_summary,
                    )

                # 5. Literature search (optional, cached -- fix #9)
                # (80% -> 90%)
                if not skip_lit:
                    try:
                        from hea_extrapolation_platform.literature_graph.search import (
                            StructuredFilter,
                        )
                        from hea_extrapolation_platform.literature_graph.feature_recommender import (
                            LiteratureFeatureRecommender,
                        )
                        log("Building literature index...")
                        yield _yield_progress(
                            "\n".join(log_lines), 85,
                            "Literature search...",
                            summary_tuple=data_summary,
                        )

                        engine = _get_literature_engine(session)
                        query = "composition only yield strength HEA"
                        sf = StructuredFilter(
                            materials_domain="HEA",
                            task="yield_strength",
                        )
                        lit_results = engine.search(
                            query, structured_filter=sf, top_n=5,
                        )
                        session["literature_results"] = lit_results

                        recommender = LiteratureFeatureRecommender(engine)
                        rec = recommender.recommend(
                            query, structured_filter=sf,
                        )
                        session["feature_recommendation"] = rec
                        log(
                            f"Literature search: "
                            f"{len(lit_results)} results"
                        )
                        yield _yield_progress(
                            "\n".join(log_lines), 90,
                            "Literature done",
                            summary_tuple=data_summary,
                        )
                    except Exception as exc:
                        log(
                            f"Literature search failed (non-fatal): {exc}"
                        )
                        yield _yield_progress(
                            "\n".join(log_lines), 90,
                            "Literature skipped (error)",
                            summary_tuple=data_summary,
                        )

                # 6. Report generation  (90% -> 100%)
                from hea_extrapolation_platform.report import (
                    ReportGenerator,
                )
                gen = ReportGenerator(out_dir=out_dir)
                best_ood = None
                best_ood_test_indices = None
                if scores and ood_results:
                    best_fs = scores[0].feature_set
                    best_ood = ood_results.get(best_fs)
                    split_info = session.get(
                        "ood_split_indices", {},
                    ).get(best_fs)
                    if split_info is not None:
                        best_ood_test_indices = split_info[1]

                report_path = gen.generate(
                    runs=runs,
                    validity_scores=scores,
                    ood_result=best_ood,
                    compositions_df=comps_df,
                    ood_test_indices=best_ood_test_indices,
                    figure_paths=figure_paths,
                    literature_results=session.get("literature_results"),
                    feature_recommendation=session.get(
                        "feature_recommendation",
                    ),
                )
                session["report_path"] = report_path
                log(f"Report: {report_path}")
                log(
                    "Analysis complete. All tabs refreshed automatically."
                )
                succeeded = True

            except Exception:
                log(f"ERROR:\n{traceback.format_exc()}")

            # --- Final yield: refresh all tabs (fix #8) ---
            if succeeded:
                final_bar = _build_progress_bar_html(100, "Complete")
            else:
                final_bar = _build_progress_bar_html(
                    last_pct, "Error \u2014 see log",
                )
            try:
                final_summary = _refresh_data_summary(session)
            except Exception:
                final_summary = empty_summary
            final_dash = _refresh_dashboard_data("rmse_test", session)
            final_res = _refresh_results_data("All", "All", "All", session)
            ood_keys = list(session.get("ood_results", {}).keys())
            ood_fs = ood_keys[0] if ood_keys else "FS_ALL"
            final_ood = _refresh_ood_data(ood_fs, session)
            final_rpt = _refresh_report_data(session)

            yield (
                final_bar,
                _build_log_html("\n".join(log_lines)),
                session,
                *final_summary,
                *final_dash,
                *final_res,
                *final_ood,
                *final_rpt,
            )

        # Wire up the run button to the generator
        run_btn.click(
            fn=run_experiment,
            inputs=[
                seeds_input, n_samples, quick_mode,
                exclude_elements, skip_literature, skip_plots,
                wf_lin_check, wf_xgb_check, wf_ens_check,
                run_csv_upload, run_csv_target,
                state,
            ],
            outputs=[
                # Config tab — progress bar + log
                progress_bar_html, progress_log, state,
                # Data Summary tab
                summary_stats_md, target_hist_plot,
                comp_bar_plot, corr_heatmap_plot,
                feature_stats_table,
                # Dashboard tab (fix #8)
                kpi_runs, kpi_best_fs, kpi_best_score, kpi_ood_count,
                integration_status, validity_plot, heatmap_plot,
                # Results tab (fix #8) — now includes interpretation + FS bar
                filter_wf, filter_fs, filter_sp,
                validity_table, results_table, parity_plot,
                results_interp_md, results_fs_bar_plot,
                # OOD tab (fix #8)
                ood_plot, ood_summary, ood_candidates,
                # Report tab (fix #8)
                report_md, download_btn,
            ],
        )

    # Fix #6: enable queue for async / streaming execution
    app.queue()

    return app


# ---------------------------------------------------------------------------
# Direct Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    application = create_app()
    # Fix #5: theme is set in gr.Blocks() constructor above
    application.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
