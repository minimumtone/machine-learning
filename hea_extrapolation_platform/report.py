"""
Report Generator for Extrapolation Discovery Platform
レポート生成モジュール

Generates a comprehensive Markdown report with embedded figures:
  - Feature validity ranking table
  - Split-wise performance comparison
  - OOD distribution figure
  - OOD candidate compositions list
  - Literature near-neighbour WF evidence (NEW)
  - Literature-derived feature recommendations (NEW)
  - Next experiment proposal

NOTE: HEA is used as a concrete example; the reporter is domain-agnostic.
All outputs are captured in a single pass (no re-execution needed).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hea_extrapolation_platform.evaluation import ValidityScore
from hea_extrapolation_platform.ood import OODResult
from hea_extrapolation_platform.workflows import RunResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate a Markdown experiment report.

    Usage::

        gen = ReportGenerator(out_dir=Path("results"))
        gen.generate(runs, scores, ood_result, ...)
    """

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = self._out_dir / "figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def figures_dir(self) -> Path:
        return self._figures_dir

    def generate(
        self,
        runs: List[RunResult],
        validity_scores: List[ValidityScore],
        ood_result: Optional[OODResult] = None,
        compositions_df: Optional[pd.DataFrame] = None,
        figure_paths: Optional[Dict[str, Path]] = None,
        extra_sections: Optional[Dict[str, str]] = None,
        literature_results: Optional[List[Any]] = None,
        feature_recommendation: Optional[Any] = None,
    ) -> Path:
        """Write full Markdown report.

        Parameters
        ----------
        runs : list of RunResult
        validity_scores : list of ValidityScore
        ood_result : OODResult, optional
        compositions_df : pd.DataFrame, optional
            Full composition table for OOD candidate listing.
        figure_paths : dict, optional
            {label: Path} for figures to embed.
        extra_sections : dict, optional
            {section_title: markdown_body} appended at end.
        literature_results : list of SearchResult, optional
            Top literature WF matches for evidence section.
        feature_recommendation : FeatureRecommendation, optional
            Literature-derived feature set recommendation.

        Returns
        -------
        Path to the generated report file.
        """
        lines: List[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ---- Header ----
        lines.append("# Extrapolation Discovery Platform - Experiment Report")
        lines.append("")
        lines.append(f"Generated: {now}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ---- 1. Summary ----
        lines.append("## 1. Experiment Summary")
        lines.append("")
        n_runs = len(runs)
        fs_names = sorted(set(r.feature_set for r in runs))
        wf_names = sorted(set(r.workflow for r in runs))
        sp_names = sorted(set(r.split_policy for r in runs))
        lines.append(f"- **Total runs**: {n_runs}")
        lines.append(f"- **Feature sets**: {', '.join(fs_names)}")
        lines.append(f"- **Workflows**: {', '.join(wf_names)}")
        lines.append(f"- **Split policies**: {', '.join(sp_names)}")
        total_time = sum(r.elapsed_sec for r in runs)
        lines.append(f"- **Total elapsed time**: {total_time:.1f} sec")
        lines.append("")

        # ---- 2. Feature Validity Ranking ----
        lines.append("## 2. Feature Set Validity Ranking")
        lines.append("")
        lines.append("| Rank | Feature Set | Effect Size | Stability | Generalisation | Leak Penalty | Extrap. Safety | **Total** |")
        lines.append("|------|-------------|-------------|-----------|----------------|--------------|----------------|-----------|")
        for i, s in enumerate(validity_scores):
            lines.append(
                f"| {i+1} | {s.feature_set} | {s.effect_size:.4f} | "
                f"{s.stability:.4f} | {s.generalisation:.4f} | "
                f"{s.leak_penalty:.4f} | {s.extrapolation_safety:.4f} | "
                f"**{s.total:.4f}** |"
            )
        lines.append("")

        # ---- 3. Performance Comparison ----
        lines.append("## 3. Split-wise Performance Comparison")
        lines.append("")
        lines.append("### RMSE (Test) by Feature Set x Split Policy")
        lines.append("")
        perf_records = []
        for r in runs:
            perf_records.append({
                "Feature Set": r.feature_set,
                "Split": r.split_policy,
                "Workflow": r.workflow,
                "RMSE_test": r.rmse_test,
                "R2_test": r.r2_test,
            })
        if perf_records:
            # Columnar construction to avoid DataFrame fragmentation
            col_names = list(perf_records[0].keys())
            df_perf = pd.DataFrame(
                {k: [r[k] for r in perf_records] for k in col_names}
            )
            pivot_rmse = df_perf.groupby(["Feature Set", "Split"])["RMSE_test"].mean().unstack(fill_value=0)
            try:
                lines.append(pivot_rmse.to_markdown())
            except ImportError:
                # tabulate not installed – fall back to plain string repr
                lines.append(pivot_rmse.to_string())
            lines.append("")
            lines.append("### R$^2$ (Test) by Feature Set x Split Policy")
            lines.append("")
            pivot_r2 = df_perf.groupby(["Feature Set", "Split"])["R2_test"].mean().unstack(fill_value=0)
            try:
                lines.append(pivot_r2.to_markdown())
            except ImportError:
                lines.append(pivot_r2.to_string())
            lines.append("")

        # ---- 4. OOD Analysis ----
        lines.append("## 4. OOD (Out-of-Distribution) Analysis")
        lines.append("")
        if ood_result is not None:
            lines.append(f"- **Total query samples**: {ood_result.n_total}")
            lines.append(f"- **OOD samples**: {ood_result.n_ood}")
            lines.append(f"- **OOD ratio**: {ood_result.ood_ratio:.2%}")
            lines.append(f"- **OOD threshold**: {ood_result.ood_threshold:.4f}")
            lines.append("")
        else:
            lines.append("OOD analysis was not performed.")
            lines.append("")

        # ---- 5. OOD Candidate Compositions ----
        lines.append("## 5. OOD Region Candidate Compositions")
        lines.append("")
        if ood_result is not None and compositions_df is not None:
            ood_mask = ood_result.is_ood
            if ood_mask.sum() > 0:
                ood_comps = compositions_df.iloc[np.where(ood_mask)[0]]
                # Show top 10 by OOD score
                ood_scores_arr = ood_result.composite_scores[ood_mask]
                sort_idx = np.argsort(-ood_scores_arr)[:10]
                top_ood = ood_comps.iloc[sort_idx]
                lines.append("Top 10 OOD candidates (highest OOD score):")
                lines.append("")
                # Format as table
                top_ood_display = top_ood.copy()
                top_ood_display["OOD_score"] = ood_scores_arr[sort_idx]
                try:
                    lines.append(top_ood_display.round(3).to_markdown())
                except ImportError:
                    lines.append(top_ood_display.round(3).to_string())
                lines.append("")
            else:
                lines.append("No OOD samples detected.")
                lines.append("")
        else:
            lines.append("Not available (OOD analysis or compositions not provided).")
            lines.append("")

        # ---- 6. Figures ----
        lines.append("## 6. Figures")
        lines.append("")
        if figure_paths:
            for label, fpath in figure_paths.items():
                rel = fpath.name
                lines.append(f"### {label}")
                lines.append("")
                lines.append(f"![{label}](figures/{rel})")
                lines.append("")
        else:
            lines.append("No figures generated.")
            lines.append("")

        # ---- 7. Literature Near-Neighbour WF Evidence ----
        lines.append("## 7. Literature Near-Neighbour WF Evidence")
        lines.append("")
        if literature_results:
            lines.append(
                "The following published workflows are most similar to the "
                "current experimental setup (ranked by embedding distance):"
            )
            lines.append("")
            lines.append(
                "| Rank | Paper ID | Model | Inputs | Split | N | "
                "Key Features | Score |"
            )
            lines.append(
                "|------|----------|-------|--------|-------|---|" 
                "-------------|-------|"
            )
            for i, sr in enumerate(literature_results):
                wf = sr.workflow
                pid = wf.paper_id
                feats = ", ".join(wf.key_features[:4])
                if len(wf.key_features) > 4:
                    feats += ", ..."
                lines.append(
                    f"| {i+1} | {pid} | {wf.model_name} | "
                    f"{wf.inputs} | {wf.split_policy} | "
                    f"{wf.data_size_n} | {feats} | "
                    f"{sr.final_score:.4f} |"
                )
            lines.append("")
        else:
            lines.append("Literature search was not performed or no results found.")
            lines.append("")

        # ---- 8. Literature-Derived Feature Recommendations ----
        lines.append("## 8. Literature-Derived Feature Recommendations")
        lines.append("")
        if feature_recommendation is not None:
            rec = feature_recommendation
            lines.append(f"**Recommended set**: {rec.name}")
            lines.append("")
            lines.append(f"- Base features ({len(rec.base_features)}): "
                         f"{', '.join(rec.base_features)}")
            if rec.added_features:
                lines.append(f"- Added from literature ({len(rec.added_features)}): "
                             f"**{', '.join(rec.added_features)}**")
            else:
                lines.append("- No additional features recommended from literature.")
            lines.append("")
            if rec.unregistered_features:
                lines.append(
                    f"Unregistered features found in literature (not added): "
                    f"{', '.join(rec.unregistered_features)}"
                )
                lines.append("")
            if rec.feature_frequency:
                lines.append("### Feature Frequency in Top Literature WFs")
                lines.append("")
                lines.append("| Feature | Count |")
                lines.append("|---------|-------|")
                for feat, count in sorted(
                    rec.feature_frequency.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"| {feat} | {count} |")
                lines.append("")
        else:
            lines.append("Literature-based feature recommendation was not performed.")
            lines.append("")

        # ---- 9. Next Experiment Proposal ----
        lines.append("## 9. Next Experiment Proposal")
        lines.append("")
        if validity_scores:
            best = validity_scores[0]
            lines.append(
                f"The best-performing feature set is **{best.feature_set}** "
                f"(total score = {best.total:.4f})."
            )
            lines.append("")
            lines.append("Recommended next steps:")
            lines.append("")
            lines.append("1. Synthesise alloys from the top OOD candidate compositions above.")
            lines.append("2. Measure yield strength experimentally.")
            lines.append(
                f"3. Re-run the platform with the updated dataset using **{best.feature_set}**."
            )
            lines.append(
                "4. Focus Block / ElementExclusion splits to verify robustness "
                "in the newly covered regions."
            )
        lines.append("")

        # ---- Extra sections ----
        if extra_sections:
            sec_num = 10
            for title, body in extra_sections.items():
                lines.append(f"## {sec_num}. {title}")
                lines.append("")
                lines.append(body)
                lines.append("")
                sec_num += 1

        # ---- Write ----
        report_path = self._out_dir / "experiment_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report written to %s", report_path)
        return report_path
