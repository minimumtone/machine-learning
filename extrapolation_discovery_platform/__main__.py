"""
CLI Entry Point for Extrapolation Discovery Platform
CLIエントリポイント

Usage::

    # Quick experiment run
    python -m extrapolation_discovery_platform run --quick --out results/

    # Full 135-run experiment
    python -m extrapolation_discovery_platform run --seeds 42 123 456 --out results/

    # Literature search
    python -m extrapolation_discovery_platform search \
        --query "composition only yield strength HEA" --top 10

    # Launch Gradio GUI
    python -m extrapolation_discovery_platform gui --port 7860

    # Re-generate report from existing registry
    python -m extrapolation_discovery_platform report \
        --registry results/run_registry.json --out results/
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Enable faulthandler so that SIGSEGV prints a Python traceback
# instead of silently crashing.  This is invaluable for diagnosing
# BLAS/LAPACK crashes caused by F-contiguous array layouts.
faulthandler.enable()

# Install pandas C-contiguous patches BEFORE importing numpy/pandas.
# This is the global SIGSEGV fix for pandas 3.0's F-contiguous layout.
from extrapolation_discovery_platform._compat import install as _install_compat
_install_compat()

import numpy as np
import pandas as pd


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-command: run  (broken into step helpers for readability)
# ---------------------------------------------------------------------------

def _run_generate_dataset(args, logger):
    """Step 1: generate or load dataset."""
    from extrapolation_discovery_platform.dataset import generate_hea_dataset

    t0 = time.time()
    comps_df, features_df, target = generate_hea_dataset(
        n_samples=args.n_samples,
        seed=args.seeds[0],
    )
    logger.info("Dataset generated in %.1f sec: %d samples, %d features",
                time.time() - t0, len(target), features_df.shape[1])
    return comps_df, features_df, target


def _run_experiment(args, comps_df, features_df, target, logger):
    """Step 2: execute experiment grid."""
    from extrapolation_discovery_platform.runner import ExperimentRunner

    runner = ExperimentRunner(
        seeds=args.seeds,
        quick=args.quick,
        exclude_elements=args.exclude_elements,
        use_mlflow=getattr(args, "use_mlflow", False),
        use_feast=getattr(args, "use_feast", False),
        use_mint=getattr(args, "use_mint", False),
    )
    runs, validity_scores, ood_results = runner.run(
        comps_df, features_df, target,
    )
    if runner.tracker.is_mlflow_active:
        logger.info("MLflow tracking URI: %s",
                    runner.tracker.get_tracking_uri())
    return runner, runs, validity_scores, ood_results


def _run_plots(runs, out_dir, logger):
    """Step 3: generate plotly figures."""
    figure_paths = {}
    try:
        from extrapolation_discovery_platform.gui.plotly_charts import (
            plotly_parity,
        )
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        best_runs = sorted(runs, key=lambda r: r.r2_test, reverse=True)
        if best_runs:
            fig = plotly_parity(best_runs[:1], title="Parity Plot (Best Model)")
            parity_path = fig_dir / "parity_best.html"
            fig.write_html(str(parity_path))
            figure_paths["Parity (best)"] = parity_path
            logger.info("Parity plot saved to %s", parity_path)
    except Exception as exc:
        logger.warning("Plotly figures failed (non-fatal): %s", exc)
    return figure_paths


def _run_literature_search(logger):
    """Step 4: optional literature search."""
    try:
        from extrapolation_discovery_platform.literature_graph.seed_data import (
            get_seed_papers, get_seed_workflows,
        )
        from extrapolation_discovery_platform.literature_graph.workflow_text import (
            generate_workflow_text,
        )
        from extrapolation_discovery_platform.literature_graph.vector_index import (
            build_index,
        )
        from extrapolation_discovery_platform.literature_graph.search import (
            LiteratureSearchEngine, StructuredFilter,
        )
        from extrapolation_discovery_platform.literature_graph.feature_recommender import (
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

        recommender = LiteratureFeatureRecommender(engine)
        feature_rec = recommender.recommend(query, structured_filter=sf)
        logger.info("Literature search complete: %d results",
                    len(lit_results))
        return lit_results, feature_rec
    except Exception as exc:
        logger.warning("Literature search failed (non-fatal): %s", exc)
        return None, None


def _run_report(
    runs, validity_scores, ood_results, runner, comps_df,
    figure_paths, literature_results, feature_rec, out_dir, logger,
):
    """Step 5: generate HTML report."""
    from extrapolation_discovery_platform.report import ReportGenerator

    best_ood = None
    best_ood_test_indices = None
    if validity_scores and ood_results:
        best_fs = validity_scores[0].feature_set
        best_ood = ood_results.get(best_fs)
        split_info = runner.ood_split_indices.get(best_fs)
        if split_info is not None:
            best_ood_test_indices = split_info[1]

    gen = ReportGenerator(out_dir=out_dir)
    return gen.generate(
        runs=runs,
        validity_scores=validity_scores,
        ood_result=best_ood,
        ood_test_indices=best_ood_test_indices,
        compositions_df=comps_df,
        figure_paths=figure_paths,
        literature_results=literature_results,
        feature_recommendation=feature_rec,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the experiment grid."""
    logger = logging.getLogger("cli.run")
    logger.info("Starting experiment: seeds=%s, n_samples=%d, quick=%s",
                args.seeds, args.n_samples, args.quick)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Dataset
    comps_df, features_df, target = _run_generate_dataset(args, logger)

    # 2. Experiment
    runner, runs, validity_scores, ood_results = _run_experiment(
        args, comps_df, features_df, target, logger,
    )
    runner.export(out_dir)

    # 3. Plots
    figure_paths = (
        _run_plots(runs, out_dir, logger) if not args.no_plots else {}
    )

    # 4. Literature
    lit_results, feature_rec = (
        _run_literature_search(logger)
        if not args.no_literature else (None, None)
    )

    # 5. Report
    report_path = _run_report(
        runs, validity_scores, ood_results, runner, comps_df,
        figure_paths, lit_results, feature_rec, out_dir, logger,
    )

    total_elapsed = time.time() - t0
    logger.info("Experiment complete: %d runs, report at %s (%.1f sec)",
                len(runs), report_path, total_elapsed)
    print(f"\nDone. {len(runs)} runs completed in {total_elapsed:.1f}s.")
    print(f"Report: {report_path}")
    print(f"Run registry: {out_dir / 'run_registry.json'}")
    if runner.tracker.is_mlflow_active:
        print(f"MLflow UI: {runner.tracker.get_tracking_uri()}")
    if runner.feature_store.is_feast_active:
        print("Feast feature store: active")
    if runner.mint_registry is not None:
        print(f"MInt workflows: {len(runner.mint_registry.list_workflows())} registered")


# ---------------------------------------------------------------------------
# Sub-command: search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    """Execute literature search."""
    from extrapolation_discovery_platform.literature_graph.seed_data import (
        get_seed_papers,
        get_seed_workflows,
    )
    from extrapolation_discovery_platform.literature_graph.workflow_text import (
        generate_workflow_text,
    )
    from extrapolation_discovery_platform.literature_graph.vector_index import build_index
    from extrapolation_discovery_platform.literature_graph.search import (
        LiteratureSearchEngine,
        StructuredFilter,
    )

    logger = logging.getLogger("cli.search")

    papers = get_seed_papers()
    workflows = get_seed_workflows()
    wf_texts = [generate_workflow_text(w) for w in workflows]
    wf_ids = [w.workflow_id for w in workflows]
    index = build_index(wf_ids, wf_texts, use_faiss=True)

    engine = LiteratureSearchEngine(
        index=index, workflows=workflows, papers=papers,
    )

    sf = StructuredFilter(
        materials_domain=args.domain,
        task=args.task,
        inputs=args.inputs,
    )

    results = engine.search(args.query, structured_filter=sf, top_n=args.top)
    logger.info("Found %d results", len(results))

    print(f"\nSearch results for: \"{args.query}\"")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        wf = r.workflow
        print(f"\n{i+1}. {wf.paper_id}")
        print(f"   Model: {wf.model_name} ({wf.model_family})")
        print(f"   Inputs: {wf.inputs}, Split: {wf.split_policy}, N={wf.data_size_n}")
        print(f"   Key features: {', '.join(wf.key_features[:5])}")
        print(f"   Score: {r.final_score:.4f}")


# ---------------------------------------------------------------------------
# Sub-command: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Re-generate report from existing run registry."""
    from extrapolation_discovery_platform.report import ReportGenerator
    from extrapolation_discovery_platform.evaluation import (
        FeatureValidityEvaluator,
        ValidityScore,
    )
    from extrapolation_discovery_platform.workflows import RunResult

    logger = logging.getLogger("cli.report")

    registry_path = Path(args.registry)
    if not registry_path.exists():
        logger.error("Registry file not found: %s", registry_path)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load runs from JSON.
    # NOTE: The JSON registry only stores scalar metrics; per-sample
    # arrays (y_test_true, y_test_pred, test_indices) and artefacts
    # (params, artifacts) are NOT persisted.  Report generation from
    # a saved registry therefore cannot produce parity plots or OOD
    # composition tables.  A warning is emitted so the user knows.
    df = pd.read_json(registry_path, orient="records")
    runs = []
    for _, row in df.iterrows():
        runs.append(RunResult(
            workflow=row["workflow"],
            feature_set=row["feature_set"],
            split_policy=row["split_policy"],
            seed=int(row["seed"]),
            fold=int(row["fold"]),
            rmse_train=float(row["rmse_train"]),
            rmse_test=float(row["rmse_test"]),
            mae_train=float(row["mae_train"]),
            mae_test=float(row["mae_test"]),
            r2_train=float(row["r2_train"]),
            r2_test=float(row["r2_test"]),
            elapsed_sec=float(row["elapsed_sec"]),
        ))
    logger.warning(
        "Loaded %d runs from JSON; y_test_true/y_test_pred/test_indices "
        "are unavailable — parity plots and OOD composition lookup will "
        "be skipped in the regenerated report.",
        len(runs),
    )

    evaluator = FeatureValidityEvaluator()
    validity_scores = evaluator.evaluate(runs)

    gen = ReportGenerator(out_dir=out_dir)
    report_path = gen.generate(runs=runs, validity_scores=validity_scores)
    logger.info("Report regenerated at %s", report_path)
    print(f"Report: {report_path}")


# ---------------------------------------------------------------------------
# Sub-command: gui
# ---------------------------------------------------------------------------

def cmd_gui(args: argparse.Namespace) -> None:
    """Launch Gradio dashboard."""
    try:
        import gradio as gr
        from extrapolation_discovery_platform.gui.app import create_app
    except ImportError as exc:
        print(f"Error: {exc}")
        print("Install Gradio with: pip install gradio plotly")
        sys.exit(1)

    app = create_app()
    # Gradio 6.0: theme moved from Blocks() to launch().
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extrapolation_discovery_platform",
        description="Extrapolation Discovery Platform — CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Execute experiment grid")
    p_run.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds (default: 42 123 456)")
    p_run.add_argument("--n-samples", type=int, default=200,
                       help="Dataset size (default: 200)")
    p_run.add_argument("--quick", action="store_true",
                       help="Use reduced HPO grids for faster execution")
    p_run.add_argument("--exclude-elements", type=str, nargs="+",
                       default=["Co", "Ni", "Ti"],
                       help="Elements for ElementExclusion splits")
    p_run.add_argument("--out", type=str, default="results",
                       help="Output directory (default: results)")
    p_run.add_argument("--no-plots", action="store_true",
                       help="Skip figure generation")
    p_run.add_argument("--no-literature", action="store_true",
                       help="Skip literature search")
    # Integration options
    p_run.add_argument("--use-mlflow", action="store_true",
                       help="Enable MLflow experiment tracking")
    p_run.add_argument("--use-feast", action="store_true",
                       help="Enable Feast feature store")
    p_run.add_argument("--use-mint", action="store_true",
                       help="Enable MInt workflow adapters")
    p_run.set_defaults(func=cmd_run)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Literature search")
    p_search.add_argument("--query", type=str, required=True,
                          help="Search query string")
    p_search.add_argument("--domain", type=str, default=None,
                          help="Materials domain filter (e.g. HEA)")
    p_search.add_argument("--task", type=str, default=None,
                          help="Task filter (e.g. yield_strength)")
    p_search.add_argument("--inputs", type=str, default=None,
                          help="Input scope filter (e.g. composition_only)")
    p_search.add_argument("--top", type=int, default=10,
                          help="Number of top results (default: 10)")
    p_search.set_defaults(func=cmd_search)

    # --- report ---
    p_report = subparsers.add_parser("report", help="Re-generate report")
    p_report.add_argument("--registry", type=str, required=True,
                          help="Path to run_registry.json")
    p_report.add_argument("--out", type=str, default="results",
                          help="Output directory")
    p_report.set_defaults(func=cmd_report)

    # --- gui ---
    p_gui = subparsers.add_parser("gui", help="Launch Gradio dashboard")
    p_gui.add_argument("--port", type=int, default=7860,
                       help="Server port (default: 7860)")
    p_gui.add_argument("--share", action="store_true",
                       help="Create a public share link")
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
