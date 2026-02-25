"""
Gradio Dashboard for Extrapolation Discovery Platform
Gradio dashboard

Launch::

    python -m hea_extrapolation_platform gui --port 7860

Tab-based layout:
  1. Dashboard  - KPI cards + validity ranking + performance heatmap
  2. Config     - Parameter UI + run button + streaming progress log
  3. Results    - Run results table + filters + parity plot
  4. OOD Map    - Interactive PCA scatter + OOD candidates table
  5. Literature - Query UI + filters + feature frequency
  6. Report     - Markdown preview + download

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
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from hea_extrapolation_platform.gui.plotly_charts import (
    plotly_feature_frequency,
    plotly_heatmap,
    plotly_ood_map,
    plotly_parity,
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


def _refresh_results_data(
    wf_filter: str, fs_filter: str, sp_filter: str,
    session: Dict[str, Any],
) -> Tuple:
    runs = session.get("runs", [])
    scores = session.get("validity_scores", [])

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

    return (
        gr.update(choices=wf_choices, value=wf_filter if wf_filter in wf_choices else "All"),
        gr.update(choices=fs_choices, value=fs_filter if fs_filter in fs_choices else "All"),
        gr.update(choices=sp_choices, value=sp_filter if sp_filter in sp_choices else "All"),
        v_df, r_df, parity_fig,
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


def _refresh_report_data(session: Dict[str, Any]) -> Tuple:
    report_path = session.get("report_path")
    if report_path is None or not Path(report_path).exists():
        return "*No report available.*", None
    content = Path(report_path).read_text(encoding="utf-8")
    return content, str(report_path)


# ---------------------------------------------------------------------------
# Main App Factory
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    # Fix #5: theme passed to gr.Blocks() constructor
    with gr.Blocks(
        title="Extrapolation Discovery Platform",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Extrapolation Discovery Platform\n"
            "Feature Validity Evaluation & OOD Detection Dashboard"
        )

        # Fix #1: per-user session state via gr.State
        state = gr.State(_empty_session)

        with gr.Tabs():
            # --- Tab 1: Dashboard ---
            with gr.Tab("Dashboard"):
                gr.Markdown("## Dashboard -- KPIs & Feature Validity")
                gr.Markdown(
                    "Run an experiment from the **Config** tab first, "
                    "then refresh this page to see results."
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

            # --- Tab 2: Config & Run ---
            with gr.Tab("Config & Run"):
                gr.Markdown("## Experiment Configuration & Execution")

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

                # Integrations are always enabled behind the scenes.
                # Users do not need to configure them.
                gr.Markdown(
                    "All experiment tracking (MLflow), feature management "
                    "(Feast), and workflow execution (MInt) are "
                    "**automatically enabled**.  "
                    "Results are recorded and managed transparently."
                )

                run_btn = gr.Button(
                    "Run Experiment", variant="primary", size="lg",
                )
                progress_log = gr.Textbox(
                    label="Progress Log",
                    lines=15,
                    interactive=False,
                )

            # --- Tab 3: Results ---
            with gr.Tab("Results"):
                gr.Markdown("## Experiment Results -- Run Table & Parity Plot")

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

            # --- Tab 4: OOD Map ---
            with gr.Tab("OOD Map"):
                gr.Markdown(
                    "## OOD (Out-of-Distribution) Map & Candidates",
                )

                fs_selector = gr.Dropdown(
                    choices=[
                        "FS_BASE", "FS_THERMO", "FS_SIZE",
                        "FS_ELECTRON", "FS_ALL",
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

                ood_refresh_btn = gr.Button(
                    "Refresh OOD Map", variant="primary",
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

            # --- Tab 5: Literature Search ---
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
                lit_recommendation = gr.Textbox(
                    label="Feature Recommendation",
                    lines=5,
                    interactive=False,
                )

                def do_search(
                    query: str, domain: str, task: str,
                    inputs_scope: str, top: float,
                    session: Dict[str, Any],
                ) -> Tuple:
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
                        r_df = (
                            pd.DataFrame(records)
                            if records
                            else pd.DataFrame()
                        )

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

                        rec_text = f"Recommended set: {rec.name}\n"
                        rec_text += (
                            f"Base features ({len(rec.base_features)}): "
                            f"{', '.join(rec.base_features)}\n"
                        )
                        if rec.added_features:
                            rec_text += (
                                f"Added from literature "
                                f"({len(rec.added_features)}): "
                                f"{', '.join(rec.added_features)}\n"
                            )
                        if rec.unregistered_features:
                            rec_text += (
                                f"Unregistered features: "
                                f"{', '.join(rec.unregistered_features)}\n"
                            )

                        return r_df, freq_fig_val, rec_text, session

                    except Exception:
                        err = traceback.format_exc()
                        return (
                            pd.DataFrame(), None,
                            f"Error:\n{err}", session,
                        )

                search_btn.click(
                    fn=do_search,
                    inputs=[
                        query_input, domain_filter, task_filter,
                        inputs_filter, lit_top_n, state,
                    ],
                    outputs=[
                        lit_results_table, freq_plot,
                        lit_recommendation, state,
                    ],
                )

            # --- Tab 6: Report ---
            with gr.Tab("Report"):
                gr.Markdown(
                    "## Experiment Report -- Markdown Preview & Download",
                )
                report_md = gr.Markdown(
                    value="*No report generated yet. "
                    "Run an experiment first.*",
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
            session: Dict[str, Any],
        ) -> Generator:
            """Generator that yields incremental progress + state.

            Using a generator enables Gradio to stream updates to the
            progress_log while the experiment runs (fix #7).
            The final yield refreshes all cross-tab components (#8).
            """
            log_lines: List[str] = []

            def log(msg: str) -> None:
                log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

            # Reset session
            session = _empty_session()

            # Placeholder outputs for cross-tab components
            empty_dash = ("0", "--", "--", "--",
                          _build_integration_status_md(None), None, None)
            empty_res = (
                gr.update(), gr.update(), gr.update(),
                pd.DataFrame(), pd.DataFrame(), None,
            )
            empty_ood = (None, "", pd.DataFrame())
            empty_rpt = ("*Running...*", None)

            def _yield_progress(log_text: str) -> Tuple:
                return (
                    log_text, session,
                    *empty_dash, *empty_res, *empty_ood, *empty_rpt,
                )

            try:
                log("Starting experiment...")
                yield _yield_progress("\n".join(log_lines))

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
                    yield _yield_progress("\n".join(log_lines))
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
                    yield _yield_progress("\n".join(log_lines))
                    return

                # Fix #2: cast slider float -> int
                n_samp_int = int(n_samp)

                # 1. Dataset generation
                from hea_extrapolation_platform.dataset import (
                    generate_hea_dataset,
                )
                log(f"Generating dataset: n={n_samp_int}, seed={seeds[0]}")
                yield _yield_progress("\n".join(log_lines))

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
                yield _yield_progress("\n".join(log_lines))

                # 2. Run experiments
                from hea_extrapolation_platform.runner import (
                    ExperimentRunner,
                )
                log(f"Running experiments: seeds={seeds}, quick={quick}")
                yield _yield_progress("\n".join(log_lines))

                runner = ExperimentRunner(
                    seeds=seeds, quick=quick, exclude_elements=excl,
                    use_mlflow=True,
                    use_feast=True,
                    use_mint=True,
                )
                runs, scores, ood_results = runner.run(
                    comps_df, features_df, target,
                )

                session["runs"] = runs
                session["validity_scores"] = scores
                session["ood_results"] = ood_results
                session["ood_split_indices"] = runner.ood_split_indices
                session["runner"] = runner

                log(f"Completed: {len(runs)} runs")

                # Log integration status (transparent to user)
                tracked = runner.tracker.list_runs()
                log(f"Experiment tracking: {len(tracked)} run(s) recorded")
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
                yield _yield_progress("\n".join(log_lines))

                # 3. Export registry  -- Fix #12: timestamp dir
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("results") / ts
                out_dir.mkdir(parents=True, exist_ok=True)
                runner.export(out_dir)
                log(
                    f"Run registry exported to "
                    f"{out_dir / 'run_registry.json'}"
                )
                yield _yield_progress("\n".join(log_lines))

                # 4. Static plots (optional)
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
                    yield _yield_progress("\n".join(log_lines))

                # 5. Literature search (optional, cached -- fix #9)
                if not skip_lit:
                    try:
                        from hea_extrapolation_platform.literature_graph.search import (
                            StructuredFilter,
                        )
                        from hea_extrapolation_platform.literature_graph.feature_recommender import (
                            LiteratureFeatureRecommender,
                        )
                        log("Building literature index...")
                        yield _yield_progress("\n".join(log_lines))

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
                        yield _yield_progress("\n".join(log_lines))
                    except Exception as exc:
                        log(
                            f"Literature search failed (non-fatal): {exc}"
                        )
                        yield _yield_progress("\n".join(log_lines))

                # 6. Report generation
                from hea_extrapolation_platform.report import (
                    ReportGenerator,
                )
                gen = ReportGenerator(out_dir=out_dir)
                best_ood = None
                if scores and ood_results:
                    best_ood = ood_results.get(scores[0].feature_set)

                report_path = gen.generate(
                    runs=runs,
                    validity_scores=scores,
                    ood_result=best_ood,
                    compositions_df=comps_df,
                    figure_paths=figure_paths,
                    literature_results=session.get("literature_results"),
                    feature_recommendation=session.get(
                        "feature_recommendation",
                    ),
                )
                session["report_path"] = report_path
                log(f"Report: {report_path}")
                log(
                    "Experiment complete. All tabs refreshed automatically."
                )

            except Exception:
                log(f"ERROR:\n{traceback.format_exc()}")

            # --- Final yield: refresh all tabs (fix #8) ---
            final_dash = _refresh_dashboard_data("rmse_test", session)
            final_res = _refresh_results_data("All", "All", "All", session)
            ood_keys = list(session.get("ood_results", {}).keys())
            ood_fs = ood_keys[0] if ood_keys else "FS_ALL"
            final_ood = _refresh_ood_data(ood_fs, session)
            final_rpt = _refresh_report_data(session)

            yield (
                "\n".join(log_lines),
                session,
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
                state,
            ],
            outputs=[
                # Config tab
                progress_log, state,
                # Dashboard tab (fix #8)
                kpi_runs, kpi_best_fs, kpi_best_score, kpi_ood_count,
                integration_status, validity_plot, heatmap_plot,
                # Results tab (fix #8)
                filter_wf, filter_fs, filter_sp,
                validity_table, results_table, parity_plot,
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
