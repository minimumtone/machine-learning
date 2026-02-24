"""
Gradio Dashboard for Extrapolation Discovery Platform
Gradioダッシュボード

Launch::

    python -m hea_extrapolation_platform gui --port 7860

Tab-based layout:
  1. Dashboard  – KPI cards + validity ranking + performance heatmap
  2. Config     – Parameter UI + run button + progress log
  3. Results    – Run results table + filters + parity plot
  4. OOD Map    – Interactive PCA scatter + OOD candidates table
  5. Literature – Query UI + filters + feature frequency
  6. Report     – Markdown preview + download
"""

from __future__ import annotations

import io
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Session State — shared across tabs
# ---------------------------------------------------------------------------

_SESSION: Dict[str, Any] = {
    "runs": [],
    "validity_scores": [],
    "ood_results": {},
    "compositions_df": None,
    "features_df": None,
    "target": None,
    "report_path": None,
    "literature_results": None,
    "feature_recommendation": None,
    "feature_counts": None,
}


def _clear_session() -> None:
    for key in _SESSION:
        if isinstance(_SESSION[key], list):
            _SESSION[key] = []
        elif isinstance(_SESSION[key], dict):
            _SESSION[key] = {}
        else:
            _SESSION[key] = None


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

def _build_dashboard_tab() -> None:
    gr.Markdown("## Dashboard — KPIs & Feature Validity")
    gr.Markdown(
        "Run an experiment from the **Config** tab first, then refresh "
        "this page to see results."
    )

    with gr.Row():
        kpi_runs = gr.Textbox(label="Total Runs", value="0", interactive=False)
        kpi_best_fs = gr.Textbox(label="Best Feature Set", value="—", interactive=False)
        kpi_best_score = gr.Textbox(label="Best Total Score", value="—", interactive=False)
        kpi_ood_count = gr.Textbox(label="OOD Samples", value="—", interactive=False)

    validity_plot = gr.Plot(label="Feature Validity Ranking")
    heatmap_plot = gr.Plot(label="Performance Heatmap (RMSE Test)")
    heatmap_metric = gr.Dropdown(
        choices=["rmse_test", "rmse_train", "mae_test", "mae_train", "r2_test", "r2_train"],
        value="rmse_test",
        label="Heatmap Metric",
    )

    def refresh_dashboard(metric: str) -> Tuple:
        runs = _SESSION["runs"]
        scores = _SESSION["validity_scores"]
        ood_results = _SESSION["ood_results"]

        n_runs = str(len(runs))
        best_fs = scores[0].feature_set if scores else "—"
        best_score = f"{scores[0].total:.4f}" if scores else "—"

        total_ood = sum(r.n_ood for r in ood_results.values()) if ood_results else 0
        ood_str = str(total_ood) if ood_results else "—"

        validity_fig = plotly_validity_ranking(scores) if scores else None
        heatmap_fig = plotly_heatmap(runs, metric=metric) if runs else None

        return n_runs, best_fs, best_score, ood_str, validity_fig, heatmap_fig

    refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
    refresh_btn.click(
        fn=refresh_dashboard,
        inputs=[heatmap_metric],
        outputs=[kpi_runs, kpi_best_fs, kpi_best_score, kpi_ood_count,
                 validity_plot, heatmap_plot],
    )
    heatmap_metric.change(
        fn=refresh_dashboard,
        inputs=[heatmap_metric],
        outputs=[kpi_runs, kpi_best_fs, kpi_best_score, kpi_ood_count,
                 validity_plot, heatmap_plot],
    )


# ---------------------------------------------------------------------------
# Tab 2: Config & Run
# ---------------------------------------------------------------------------

def _build_config_tab() -> None:
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
                info="Use reduced hyperparameter grids for faster execution",
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
                info="Skip PNG generation (Plotly charts are always available)",
            )

    run_btn = gr.Button("Run Experiment", variant="primary", size="lg")
    progress_log = gr.Textbox(
        label="Progress Log",
        lines=15,
        interactive=False,
    )

    def run_experiment(
        seeds_str: str,
        n_samp: int,
        quick: bool,
        excl_str: str,
        skip_lit: bool,
        skip_plt: bool,
    ) -> str:
        log_lines: List[str] = []

        def log(msg: str) -> None:
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        try:
            _clear_session()
            log("Starting experiment...")

            seeds = [int(s.strip()) for s in seeds_str.split() if s.strip()]
            excl = [e.strip() for e in excl_str.split() if e.strip()]

            if not seeds:
                return "Error: No valid seeds provided."

            # 1. Dataset generation
            from hea_extrapolation_platform.dataset import generate_hea_dataset
            log(f"Generating dataset: n={n_samp}, seed={seeds[0]}")
            comps_df, features_df, target = generate_hea_dataset(
                n_samples=n_samp, seed=seeds[0],
            )
            _SESSION["compositions_df"] = comps_df
            _SESSION["features_df"] = features_df
            _SESSION["target"] = target
            log(f"Dataset: {len(target)} samples, {features_df.shape[1]} features")

            # 2. Run experiments
            from hea_extrapolation_platform.runner import ExperimentRunner
            log(f"Running experiments: seeds={seeds}, quick={quick}")
            runner = ExperimentRunner(
                seeds=seeds, quick=quick, exclude_elements=excl,
            )
            runs, scores, ood_results = runner.run(comps_df, features_df, target)

            _SESSION["runs"] = runs
            _SESSION["validity_scores"] = scores
            _SESSION["ood_results"] = ood_results

            log(f"Completed: {len(runs)} runs")
            if scores:
                log(f"Best feature set: {scores[0].feature_set} "
                    f"(score={scores[0].total:.4f})")

            for fs_key, ood_res in ood_results.items():
                log(f"OOD [{fs_key}]: {ood_res.n_ood}/{ood_res.n_total} "
                    f"({ood_res.ood_ratio:.1%})")

            # 3. Export registry
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            runner.export(out_dir)
            log(f"Run registry exported to {out_dir / 'run_registry.json'}")

            # 4. Static plots (optional)
            figure_paths: Dict[str, Path] = {}
            if not skip_plt:
                from hea_extrapolation_platform.visualization import (
                    plot_validity_ranking,
                    plot_performance_heatmap,
                    plot_parity,
                )
                fig_dir = out_dir / "figures"
                figure_paths["Validity"] = plot_validity_ranking(scores, fig_dir)
                figure_paths["Heatmap"] = plot_performance_heatmap(runs, fig_dir)
                figure_paths["Parity"] = plot_parity(runs, fig_dir)
                log("Static plots saved.")

            # 5. Literature search (optional)
            if not skip_lit:
                try:
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
                        LiteratureSearchEngine, StructuredFilter,
                    )
                    from hea_extrapolation_platform.literature_graph.feature_recommender import (
                        LiteratureFeatureRecommender,
                    )

                    papers = get_seed_papers()
                    workflows = get_seed_workflows()
                    wf_texts = [generate_workflow_text(w) for w in workflows]
                    wf_ids = [w.workflow_id for w in workflows]
                    index = build_index(wf_ids, wf_texts, use_faiss=True)

                    engine = LiteratureSearchEngine(
                        index=index, workflows=workflows, papers=papers,
                    )
                    query = "composition only yield strength HEA"
                    sf = StructuredFilter(materials_domain="HEA", task="yield_strength")
                    lit_results = engine.search(query, structured_filter=sf, top_n=5)
                    _SESSION["literature_results"] = lit_results

                    recommender = LiteratureFeatureRecommender(engine)
                    rec = recommender.recommend(query, structured_filter=sf)
                    _SESSION["feature_recommendation"] = rec
                    log(f"Literature search: {len(lit_results)} results found")
                except Exception as exc:
                    log(f"Literature search failed (non-fatal): {exc}")

            # 6. Report generation
            from hea_extrapolation_platform.report import ReportGenerator
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
                literature_results=_SESSION.get("literature_results"),
                feature_recommendation=_SESSION.get("feature_recommendation"),
            )
            _SESSION["report_path"] = report_path
            log(f"Report: {report_path}")

            log("Experiment complete. Switch to other tabs to explore results.")

        except Exception:
            log(f"ERROR:\n{traceback.format_exc()}")

        return "\n".join(log_lines)

    run_btn.click(
        fn=run_experiment,
        inputs=[seeds_input, n_samples, quick_mode, exclude_elements,
                skip_literature, skip_plots],
        outputs=[progress_log],
    )


# ---------------------------------------------------------------------------
# Tab 3: Results
# ---------------------------------------------------------------------------

def _build_results_tab() -> None:
    gr.Markdown("## Experiment Results — Run Table & Parity Plot")

    with gr.Row():
        filter_wf = gr.Dropdown(
            choices=["All", "WF-LIN", "WF-XGB", "WF-ENS"],
            value="All",
            label="Workflow Filter",
        )
        filter_fs = gr.Dropdown(
            choices=["All", "FS_BASE", "FS_THERMO",
                     "FS_SIZE", "FS_ELECTRON", "FS_ALL"],
            value="All",
            label="Feature Set Filter",
        )
        filter_sp = gr.Dropdown(
            choices=["All", "RandomCV", "CompositionBlock", "ElementExclusion"],
            value="All",
            label="Split Policy Filter",
        )

    validity_table = gr.Dataframe(label="Feature Validity Ranking")
    results_table = gr.Dataframe(label="Run Results")
    parity_plot = gr.Plot(label="Parity Plot")

    def refresh_results(wf_filter: str, fs_filter: str, sp_filter: str) -> Tuple:
        runs = _SESSION["runs"]
        scores = _SESSION["validity_scores"]

        # Validity table
        v_df = validity_scores_to_dataframe(scores) if scores else pd.DataFrame()

        # Apply filters
        filtered = runs
        if wf_filter != "All":
            filtered = [r for r in filtered if r.workflow == wf_filter]
        if fs_filter != "All":
            filtered = [r for r in filtered if r.feature_set == fs_filter]
        if sp_filter != "All":
            filtered = [r for r in filtered if r.split_policy == sp_filter]

        r_df = runs_to_dataframe(filtered) if filtered else pd.DataFrame()
        parity_fig = plotly_parity(filtered) if filtered else None

        return v_df, r_df, parity_fig

    refresh_btn = gr.Button("Refresh Results", variant="primary")
    refresh_btn.click(
        fn=refresh_results,
        inputs=[filter_wf, filter_fs, filter_sp],
        outputs=[validity_table, results_table, parity_plot],
    )
    # Auto-refresh on filter change
    for dropdown in [filter_wf, filter_fs, filter_sp]:
        dropdown.change(
            fn=refresh_results,
            inputs=[filter_wf, filter_fs, filter_sp],
            outputs=[validity_table, results_table, parity_plot],
        )


# ---------------------------------------------------------------------------
# Tab 4: OOD Map
# ---------------------------------------------------------------------------

def _build_ood_tab() -> None:
    gr.Markdown("## OOD (Out-of-Distribution) Map & Candidates")

    fs_selector = gr.Dropdown(
        choices=["FS_BASE", "FS_THERMO",
                 "FS_SIZE", "FS_ELECTRON", "FS_ALL"],
        value="FS_ALL",
        label="Feature Set for OOD Map",
    )

    ood_plot = gr.Plot(label="OOD Map (PCA)")

    with gr.Row():
        ood_summary = gr.Textbox(label="OOD Summary", interactive=False)

    ood_candidates = gr.Dataframe(label="Top OOD Candidates")

    def refresh_ood(fs_key: str) -> Tuple:
        ood_results = _SESSION["ood_results"]
        features_df = _SESSION["features_df"]
        comps_df = _SESSION["compositions_df"]

        if not ood_results or features_df is None:
            return None, "No OOD results. Run experiment first.", pd.DataFrame()

        ood_res = ood_results.get(fs_key)
        if ood_res is None:
            available = list(ood_results.keys())
            return (None,
                    f"No OOD for {fs_key}. Available: {available}",
                    pd.DataFrame())

        # Get feature columns for the selected set
        from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
        try:
            fs_enum = FeatureSetName(fs_key)
            cols = FeatureCatalog.columns(fs_enum)
        except (ValueError, KeyError):
            return None, f"Unknown feature set: {fs_key}", pd.DataFrame()

        X_all = features_df[cols]
        # Split based on last fold (use full dataset as both train & query for vis)
        # In reality, we'd want the actual train/test split; for now,
        # we visualize the entire dataset with OOD scores overlaid.
        n_total = len(X_all)
        n_ood_scores = len(ood_res.composite_scores)

        # If OOD scores cover only a subset, we need to handle this
        if n_ood_scores < n_total:
            X_train = X_all.iloc[:n_total - n_ood_scores]
            X_query = X_all.iloc[n_total - n_ood_scores:]
        else:
            # Scores cover full dataset
            X_train = X_all.iloc[:n_total // 2]
            X_query = X_all.iloc[n_total // 2:]
            # Trim OOD scores to match query size
            ood_scores_vis = ood_res.composite_scores[:len(X_query)]
            is_ood_vis = ood_res.is_ood[:len(X_query)]

        if n_ood_scores < n_total:
            ood_scores_vis = ood_res.composite_scores
            is_ood_vis = ood_res.is_ood

        fig = plotly_ood_map(
            X_train, X_query,
            composite_scores=ood_scores_vis,
            is_ood=is_ood_vis,
            title=f"OOD Map (PCA) — {fs_key}",
        )

        summary = (
            f"Total query: {ood_res.n_total} | "
            f"OOD: {ood_res.n_ood} ({ood_res.ood_ratio:.1%}) | "
            f"Threshold: {ood_res.ood_threshold:.4f}"
        )

        # OOD candidates table
        cand_df = pd.DataFrame()
        if comps_df is not None and ood_res.is_ood.any():
            ood_mask = ood_res.is_ood
            # Map OOD indices back to composition DataFrame
            if n_ood_scores <= len(comps_df):
                offset = len(comps_df) - n_ood_scores
                ood_global_idx = np.where(ood_mask)[0] + offset
                ood_global_idx = ood_global_idx[ood_global_idx < len(comps_df)]
                if len(ood_global_idx) > 0:
                    cand_df = comps_df.iloc[ood_global_idx].copy()
                    cand_df["OOD_Score"] = ood_res.composite_scores[
                        ood_global_idx - offset
                    ]
                    cand_df = cand_df.sort_values("OOD_Score", ascending=False).head(20)
                    cand_df = cand_df.round(3)

        return fig, summary, cand_df

    refresh_btn = gr.Button("Refresh OOD Map", variant="primary")
    refresh_btn.click(
        fn=refresh_ood,
        inputs=[fs_selector],
        outputs=[ood_plot, ood_summary, ood_candidates],
    )
    fs_selector.change(
        fn=refresh_ood,
        inputs=[fs_selector],
        outputs=[ood_plot, ood_summary, ood_candidates],
    )


# ---------------------------------------------------------------------------
# Tab 5: Literature Search
# ---------------------------------------------------------------------------

def _build_literature_tab() -> None:
    gr.Markdown("## Literature Search — Embedding + Structured Filters")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Search Query",
                value="composition only yield strength HEA",
                lines=2,
                info="Natural language or workflow-text-like query",
            )
        with gr.Column(scale=1):
            domain_filter = gr.Textbox(label="Domain", value="HEA")
            task_filter = gr.Textbox(label="Task", value="yield_strength")

    with gr.Row():
        inputs_filter = gr.Dropdown(
            choices=["", "composition_only", "+process", "+calphad", "+microstructure"],
            value="",
            label="Inputs Scope",
        )
        top_n = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Top N")

    search_btn = gr.Button("Search Literature", variant="primary")

    results_table = gr.Dataframe(label="Search Results")
    freq_plot = gr.Plot(label="Feature Frequency")
    recommendation = gr.Textbox(label="Feature Recommendation", lines=5, interactive=False)

    def do_search(
        query: str, domain: str, task: str,
        inputs_scope: str, top: int,
    ) -> Tuple:
        try:
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
                LiteratureSearchEngine, StructuredFilter,
            )
            from hea_extrapolation_platform.literature_graph.feature_recommender import (
                LiteratureFeatureRecommender,
            )

            papers = get_seed_papers()
            workflows = get_seed_workflows()
            wf_texts = [generate_workflow_text(w) for w in workflows]
            wf_ids = [w.workflow_id for w in workflows]
            index = build_index(wf_ids, wf_texts, use_faiss=True)

            engine = LiteratureSearchEngine(
                index=index, workflows=workflows, papers=papers,
            )

            sf = StructuredFilter(
                materials_domain=domain if domain.strip() else None,
                task=task if task.strip() else None,
                inputs=inputs_scope if inputs_scope.strip() else None,
            )

            results = engine.search(query, structured_filter=sf, top_n=top)
            _SESSION["literature_results"] = results

            # Results table
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
                    "Key Features": ", ".join(wf.key_features[:5]),
                    "Score": round(r.final_score, 4),
                })
            r_df = pd.DataFrame(records) if records else pd.DataFrame()

            # Feature frequency
            _, feature_counts = engine.search_for_features(
                query, structured_filter=sf, top_n=top,
            )
            _SESSION["feature_counts"] = feature_counts
            freq_fig = plotly_feature_frequency(feature_counts)

            # Feature recommendation
            recommender = LiteratureFeatureRecommender(engine)
            rec = recommender.recommend(query, structured_filter=sf)
            _SESSION["feature_recommendation"] = rec

            rec_text = f"Recommended set: {rec.name}\n"
            rec_text += f"Base features ({len(rec.base_features)}): {', '.join(rec.base_features)}\n"
            if rec.added_features:
                rec_text += f"Added from literature ({len(rec.added_features)}): {', '.join(rec.added_features)}\n"
            if rec.unregistered_features:
                rec_text += f"Unregistered features: {', '.join(rec.unregistered_features)}\n"

            return r_df, freq_fig, rec_text

        except Exception:
            err = traceback.format_exc()
            return pd.DataFrame(), None, f"Error:\n{err}"

    search_btn.click(
        fn=do_search,
        inputs=[query_input, domain_filter, task_filter, inputs_filter, top_n],
        outputs=[results_table, freq_plot, recommendation],
    )


# ---------------------------------------------------------------------------
# Tab 6: Report
# ---------------------------------------------------------------------------

def _build_report_tab() -> None:
    gr.Markdown("## Experiment Report — Markdown Preview & Download")

    report_md = gr.Markdown(value="*No report generated yet. Run an experiment first.*")
    download_btn = gr.File(label="Download Report (.md)")

    def refresh_report() -> Tuple:
        report_path = _SESSION.get("report_path")
        if report_path is None or not Path(report_path).exists():
            return "*No report available.*", None

        content = Path(report_path).read_text(encoding="utf-8")
        return content, str(report_path)

    refresh_btn = gr.Button("Refresh Report", variant="primary")
    refresh_btn.click(
        fn=refresh_report,
        inputs=[],
        outputs=[report_md, download_btn],
    )


# ---------------------------------------------------------------------------
# Main App Factory
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        title="Extrapolation Discovery Platform",
    ) as app:
        gr.Markdown(
            "# Extrapolation Discovery Platform\n"
            "外挿発見基盤 — Feature Validity Evaluation & OOD Detection Dashboard"
        )

        with gr.Tabs():
            with gr.Tab("Dashboard"):
                _build_dashboard_tab()
            with gr.Tab("Config & Run"):
                _build_config_tab()
            with gr.Tab("Results"):
                _build_results_tab()
            with gr.Tab("OOD Map"):
                _build_ood_tab()
            with gr.Tab("Literature Search"):
                _build_literature_tab()
            with gr.Tab("Report"):
                _build_report_tab()

    return app


# ---------------------------------------------------------------------------
# Direct Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
