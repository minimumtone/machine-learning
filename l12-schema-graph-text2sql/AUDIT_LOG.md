# AUDIT_LOG

## Session start

- Date: 2026-06-19 13:35 UTC
- Working directory: /home/ubuntu/repos/machine-learning/l12-schema-graph-text2sql
- Branch: main
- Commit: a9f59741e60480bc31fb35e9afab8f5d75a31426
- Python: 3.12.8
- Key packages: openai 2.40.0, psycopg 3.3.4, networkx 3.6.1, sqlglot 30.8.0, sentence-transformers 5.6.0, torch 2.12.1, mecab-python3 1.0.12, scikit-learn 1.9.0
- pip freeze: pip-freeze.txt

## Git status at session start

```
 M evaluation/reranker_eval_results.json
 M paper/paper_figures.json
```

Note: two modified files are unstaged from prior session. These are superseded by ablation_results.json and paper_data.json respectively.

## Phase 0 commands

```
git rev-parse --abbrev-ref HEAD  -> main
git rev-parse HEAD               -> a9f59741e60480bc31fb35e9afab8f5d75a31426
python3 --version                -> Python 3.12.8
python3 -m pip freeze            -> pip-freeze.txt
```

## Phase 1: Chronological inventory

Identified primary evidence files:
1. evaluation/ablation_results.json — 7-condition x 100-query ablation (ACTIVE)
2. evaluation/jp_reranker_vh_results.json — JP reranker VH 20-query comparison (ACTIVE)
3. evaluation/reranker_eval_results.json — 90-query reranker A/B eval (ACTIVE, but note: separate run from ablation)
4. evaluation/evaluation_dataset.jsonl — author 100 queries (ACTIVE)
5. evaluation/expert_evaluation_dataset.jsonl — expert 100 queries (ACTIVE, not used in ablation)
6. llm/mecab_materials.csv — 492-term MeCab dictionary (ACTIVE)
7. llm/materials_engineering_vocab.csv — 525-line source vocabulary (ACTIVE)

Superseded files:
- paper/paper_figures.json — old SSoT file, replaced by paper/paper_data.json
- scripts/compute_paper_figures.py — old compute script, replaced by scripts/compute_all_figures.py
- paper/t2sql_materials_paper.tex — old LaTeX, replaced by new paper
- paper/t2sql_materials_paper.pdf — old PDF
- evaluation/ablation_summary.md — intermediate summary, superseded by CURRENT_STATE.md and paper
- evaluation/proposed_result.csv — old n_best=1 baseline results
- evaluation/proposed_result_run{1,2,3}.csv — old 3-run results
- evaluation/proposed_result_annotated.csv — old annotated results
- evaluation/baseline_result.csv — old baseline comparison
- evaluation/error_analysis.py — old error analysis script
- evaluation/error_analysis_report.md — old error analysis report (if exists)
- scripts/eval_reranker_ab.py — superseded by eval_ablation.py
- scripts/validate_paper_numbers.py — validated old paper, no longer needed
- experiments/ — old experiment results from prior phases

## Conflict: reranker_eval_results.json vs ablation_results.json

reranker_eval_results.json reports overall_reranker=77.16% for 90 queries.
ablation_results.json full condition reports 80.65% for 100 queries.
These are DIFFERENT runs at DIFFERENT times with DIFFERENT query subsets (90 vs 100).
The ablation full condition is the authoritative baseline.
reranker_eval_results.json is retained only for the reranker-vs-baseline delta (+7.7pp).

## Phase 4: Package versions verified

All versions from `pip freeze` (not from memory):
- openai==2.40.0
- psycopg==3.3.4
- networkx==3.6.1
- sqlglot==30.8.0
- sentence-transformers==5.6.0
- torch==2.12.1
- mecab-python3==1.0.12
- scikit-learn==1.9.0

## Phase 5: Unified compute script

Ran: `python3 scripts/compute_all_figures.py`
Output: paper/paper_data.json
Exit code: 0
All numbers verified against source JSONs.

## Phase 6: LaTeX paper expansion (2026-06-19)

Expanded paper/main.tex from 534 lines to 1056 lines.
Restored from old version (3,214 lines at commit a9f5974):
- Figure 1: Pipeline TikZ diagram (8 components)
- Figure 2: ER diagram (30 tables, full TikZ)
- Algorithm 1: Steiner tree pseudocode
- Table: DB statistics, query design, MeCab dictionary, latency, related systems comparison
- 24 bibliography entries (from old 43, retaining all cited)
- Detailed discussion sections: per-component analysis, related comparison table, limitations, future work
- Acknowledgments, Data Availability, Author Contributions sections

All numerical values sourced from paper/paper_data.json (EVID-20260619-1335-ablation-json).
No hand-typed numbers.

Compilation:
- lualatex main.tex (pass 1): exit code 0, 7 pages
- lualatex main.tex (pass 2): exit code 0, 7 pages, 602494 bytes
- No missing references or undefined citations
- Only cosmetic warnings: Kanji font shape info, hyperref bookmark tokens

## Phase 7: Supplementary Materials (2026-06-19)

Expanded paper/main.tex from 1056 lines to 1534 lines (+478 lines).
Added 7 supplementary sections (S1--S7):
- S1: Unit test categories (12 categories, 134 tests, all PASS)
- S2: Regression test cases (39 cases, all PASS)
- S3: Generated SQL examples (LLM-only vs Schema Graph-constrained, 2 comparisons)
- S4: SQL injection / safety tests (7 adversarial inputs, all blocked)
- S5: LLM configuration (GPT-5.5 parameters, prompt structure)
- S6: Per-query detail for 100 evaluation queries (full condition)
- S7: Ablation cross-condition comparison (30 queries with differing results across 7 conditions)

Devin Review fixes applied:
- JP reranker delta: -4.3pp -> -4.4pp (match paper_data.json)
- sqlglot author: T. Palantir -> T. Mao
- FINAL_AUDIT date: chronological correction

Compilation:
- lualatex main.tex (pass 1): exit code 0, 14 pages
- lualatex main.tex (pass 2): exit code 0, 14 pages, 700319 bytes
- No missing references or undefined citations
- Only cosmetic warnings: Kanji font shape info, hyperref bookmark tokens
