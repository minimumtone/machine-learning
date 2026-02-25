"""
CLI Entry Point for Extrapolation Discovery Platform
CLIエントリポイント

Usage::

    # Quick experiment run
    python -m hea_extrapolation_platform run --quick --out results/

    # Full 135-run experiment
    python -m hea_extrapolation_platform run --seeds 42 123 456 --out results/

    # Literature search
    python -m hea_extrapolation_platform search \
        --query "composition only yield strength HEA" --top 10

    # Launch Gradio GUI
    python -m hea_extrapolation_platform gui --port 7860

    # Re-generate report from existing registry
    python -m hea_extrapolation_platform report \
        --registry results/run_registry.json --out results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

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
# Sub-command: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Execute the experiment grid."""
    from hea_extrapolation_platform.dataset import generate_hea_dataset
    from hea_extrapolation_platform.runner import ExperimentRunner
    from hea_extrapolation_platform.visualization import (
        plot_ood_map_pca,
        plot_validity_ranking,
        plot_performance_heatmap,
        plot_parity,
        plot_uncertainty_vs_ood,
    )
    from hea_extrapolation_platform.report import ReportGenerator

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    seeds: List[int] = args.seeds
    logger = logging.getLogger("cli.run")
    logger.info("Starting experiment: seeds=%s, n_samples=%d, quick=%s",
                seeds, args.n_samples, args.quick)

    # Integration flags
    use_mlflow = getattr(args, "use_mlflow", False)
    use_feast = getattr(args, "use_feast", False)
    use_mint = getattr(args, "use_mint", False)

    if use_mlflow:
        logger.info("MLflow tracking enabled")
    if use_feast:
        logger.info("Feast feature store enabled")
    if use_mint:
        logger.info("MInt workflow adapters enabled")

    # 1. Dataset generation
    t0 = time.time()
    comps_df, features_df, target = generate_hea_dataset(
        n_samples=args.n_samples,
        seed=seeds[0],
    )
    logger.info("Dataset generated in %.1f sec: %d samples, %d features",
                time.time() - t0, len(target), features_df.shape[1])

    # 2. Experiment execution (with optional integrations)
    runner = ExperimentRunner(
        seeds=seeds,
        quick=args.quick,
        exclude_elements=args.exclude_elements,
        use_mlflow=use_mlflow,
        use_feast=use_feast,
        use_mint=use_mint,
    )
    runs, validity_scores, ood_results = runner.run(comps_df, features_df, target)

    # Log integration status
    if runner.tracker.is_mlflow_active:
        logger.info("MLflow tracking URI: %s", runner.tracker.get_tracking_uri())

    # 3. Export run registry
    runner.export(out_dir)

    # 4. Visualisation
    figure_paths = {}
    if not args.no_plots:
        logger.info("Generating figures...")
        figure_paths["Validity Ranking"] = plot_validity_ranking(
            validity_scores, fig_dir,
        )
        figure_paths["Performance Heatmap"] = plot_performance_heatmap(
            runs, fig_dir,
        )
        figure_paths["Parity Plot"] = plot_parity(runs, fig_dir)

        # OOD maps per feature set — use actual train/test split indices
        # stored by the runner to avoid size mismatch with ood_res.is_ood.
        for fs_key, ood_res in ood_results.items():
            from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
            try:
                fs_enum = FeatureSetName(fs_key)
                cols = FeatureCatalog.columns(fs_enum)
                split_indices = runner.ood_split_indices.get(fs_key)
                if split_indices is None:
                    logger.warning("No OOD split indices for %s, skipping plot", fs_key)
                    continue
                train_idx_vis, test_idx_vis = split_indices
                X_train_vis = features_df[cols].iloc[train_idx_vis]
                X_test_vis = features_df[cols].iloc[test_idx_vis]
                fig_path = plot_ood_map_pca(
                    X_train_vis, X_test_vis, ood_res, fig_dir,
                    filename=f"ood_map_pca_{fs_key}.png",
                    title=f"OOD Map (PCA) - {fs_key}",
                )
                figure_paths[f"OOD Map - {fs_key}"] = fig_path
            except Exception as exc:
                logger.warning("Failed to plot OOD map for %s: %s", fs_key, exc)

    # 5. Literature search (optional)
    literature_results = None
    feature_rec = None
    if not args.no_literature:
        try:
            from hea_extrapolation_platform.literature_graph.seed_data import (
                get_seed_papers,
                get_seed_workflows,
            )
            from hea_extrapolation_platform.literature_graph.workflow_text import (
                generate_workflow_text,
            )
            from hea_extrapolation_platform.literature_graph.vector_index import (
                build_index,
            )
            from hea_extrapolation_platform.literature_graph.search import (
                LiteratureSearchEngine,
                StructuredFilter,
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
            literature_results = engine.search(query, structured_filter=sf, top_n=5)

            recommender = LiteratureFeatureRecommender(engine)
            feature_rec = recommender.recommend(query, structured_filter=sf)
            logger.info("Literature search complete: %d results", len(literature_results))
        except Exception as exc:
            logger.warning("Literature search failed (non-fatal): %s", exc)

    # 6. Report generation
    gen = ReportGenerator(out_dir=out_dir)
    # Best OOD result for report
    best_ood = None
    if validity_scores and ood_results:
        best_fs = validity_scores[0].feature_set
        best_ood = ood_results.get(best_fs)

    report_path = gen.generate(
        runs=runs,
        validity_scores=validity_scores,
        ood_result=best_ood,
        compositions_df=comps_df,
        figure_paths=figure_paths,
        literature_results=literature_results,
        feature_recommendation=feature_rec,
    )

    total_elapsed = time.time() - t0
    logger.info(
        "Experiment complete: %d runs, report at %s (%.1f sec total)",
        len(runs), report_path, total_elapsed,
    )
    print(f"\nDone. {len(runs)} runs completed in {total_elapsed:.1f}s.")
    print(f"Report: {report_path}")
    print(f"Run registry: {out_dir / 'run_registry.json'}")

    # Print integration status
    if runner.tracker.is_mlflow_active:
        print(f"MLflow UI: {runner.tracker.get_tracking_uri()}")
    if runner.feature_store.is_feast_active:
        print("Feast feature store: active")
    if runner.mint_registry is not None:
        n_mint = len(runner.mint_registry.list_workflows())
        print(f"MInt workflows: {n_mint} registered")


# ---------------------------------------------------------------------------
# Sub-command: search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    """Execute literature search."""
    from hea_extrapolation_platform.literature_graph.seed_data import (
        get_seed_papers,
        get_seed_workflows,
    )
    from hea_extrapolation_platform.literature_graph.workflow_text import (
        generate_workflow_text,
    )
    from hea_extrapolation_platform.literature_graph.vector_index import build_index
    from hea_extrapolation_platform.literature_graph.search import (
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
    from hea_extrapolation_platform.report import ReportGenerator
    from hea_extrapolation_platform.evaluation import (
        FeatureValidityEvaluator,
        ValidityScore,
    )
    from hea_extrapolation_platform.workflows import RunResult

    logger = logging.getLogger("cli.report")

    registry_path = Path(args.registry)
    if not registry_path.exists():
        logger.error("Registry file not found: %s", registry_path)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load runs from JSON
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
        from hea_extrapolation_platform.gui.app import create_app
    except ImportError as exc:
        print(f"Error: {exc}")
        print("Install Gradio with: pip install gradio plotly")
        sys.exit(1)

    app = create_app()
    # theme is set in gr.Blocks() inside create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hea_extrapolation_platform",
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
