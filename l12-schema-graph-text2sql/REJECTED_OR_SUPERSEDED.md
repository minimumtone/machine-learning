# REJECTED_OR_SUPERSEDED

## Rejected parameters

| Parameter | Value | Reason | Evidence ID |
|-----------|-------|--------|-------------|
| gpt-4o-mini | N/A | User explicitly prohibited | N/A |

## Superseded parameters

| Parameter | Old value | New value | Evidence ID |
|-----------|-----------|-----------|-------------|
| n_best | 1 | 3 | EVID-20260619-1335-ablation-json |
| SSoT file | paper/paper_figures.json | paper/paper_data.json | EVID-20260619-1335-paper-data |
| Compute script | scripts/compute_paper_figures.py | scripts/compute_all_figures.py | EVID-20260619-1335-paper-data |

## Failed runs

None in current session. Prior session had OpenAI content filter rejection at q_hard_030 during no_fewshot condition, handled by try/except wrapper.

## Rejected results

| Result | Value | Reason |
|--------|-------|--------|
| Old proposed_result.csv overall | ~70.8% | From n_best=1 baseline, superseded by ablation full (80.6%) |

## Superseded results

| Result | Old value | New value | Evidence ID |
|--------|-----------|-----------|-------------|
| Pipeline accuracy | ~70.8% (proposed_result.csv) | 80.6% (ablation full) | EVID-20260619-1335-ablation-full |
| 20-query reranker delta | +18.9pp | +7.7pp (full 90q) | EVID-20260619-1335-reranker-eval |

## Draft notes not allowed in final report

- evaluation/ablation_summary.md — intermediate summary, all values now in paper_data.json

## Do-not-use items

| Item | Reason |
|------|--------|
| gpt-4o-mini | User prohibited |
| paper/paper_figures.json | Old SSoT, superseded |
| paper/t2sql_materials_paper.tex | Old LaTeX, to be deleted |
| paper/t2sql_materials_paper.pdf | Old PDF, to be deleted |
| scripts/compute_paper_figures.py | Old compute script, superseded |
| scripts/validate_paper_numbers.py | Validated old paper only |
| evaluation/proposed_result.csv | Old n_best=1 results |
| evaluation/proposed_result_run{1,2,3}.csv | Old 3-run results |
| evaluation/proposed_result_annotated.csv | Old annotated results |
| evaluation/baseline_result.csv | Old baseline comparison |
| evaluation/error_analysis.py | Old error analysis |
| evaluation/error_analysis_report.md | Old error analysis report |
| scripts/eval_reranker_ab.py | Superseded by eval_ablation.py |
| evaluation/expert_evaluation_results.json | Old expert eval, not used in ablation |
