# CURRENT_STATE

## Current objective

Create a new LaTeX paper and PDF for "Graph-Constrained Text-to-SQL Pipeline for Inorganic Materials Databases" with all numerical values sourced from `scripts/compute_all_figures.py` via `paper/paper_data.json`.

## Current data sources

- `evaluation/ablation_results.json` — 7-condition x 100-query ablation [EVID-20260619-1335-ablation-json]
- `evaluation/jp_reranker_vh_results.json` — JP reranker VH 20 [EVID-20260619-1335-jp-reranker]
- `evaluation/reranker_eval_results.json` — 90-query reranker A/B [EVID-20260619-1335-reranker-eval]
- `evaluation/evaluation_dataset.jsonl` — 100 author queries [EVID-20260619-1335-dataset-author]
- `llm/mecab_materials.csv` — 492-term MeCab dictionary [EVID-20260619-1335-mecab-dict]
- `paper/paper_data.json` — unified SSoT output [EVID-20260619-1335-paper-data]

## Current valid parameters

| Parameter | Value | Evidence ID |
|-----------|-------|-------------|
| LLM model | gpt-5.5 | EVID-20260619-1335-model-gpt55 |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | EVID-20260619-1335-ablation-json |
| n_best | 3 | EVID-20260619-1335-ablation-json |
| Author queries | 100 (E=20, M=30, H=30, VH=20) | EVID-20260619-1335-dataset-author |
| DB tables | 30 | EVID-20260619-1335-db-schema |
| Material entries | 1,470 | EVID-20260619-1335-db-schema |
| MeCab terms | 492 | EVID-20260619-1335-mecab-dict |
| Ablation conditions | 7 | EVID-20260619-1335-ablation-json |
| Total evaluations | 700 | EVID-20260619-1335-ablation-json |

## Current valid results

| Result | Value | Evidence ID |
|--------|-------|-------------|
| Full pipeline accuracy | 80.6% | EVID-20260619-1335-ablation-full |
| no_fewshot delta | -12.5pp | EVID-20260619-1335-ablation-nofewshot |
| no_dict delta | -6.7pp | EVID-20260619-1335-ablation-nodict |
| no_reranker delta | -4.1pp | EVID-20260619-1335-ablation-noreranker |
| no_guard delta | -0.2pp | EVID-20260619-1335-ablation-noguard |
| no_nbest delta | -0.2pp | EVID-20260619-1335-ablation-nonbest |
| no_graph delta | -0.2pp | EVID-20260619-1335-ablation-nograph |
| Reranker A/B delta (90q) | +7.7pp | EVID-20260619-1335-reranker-eval |
| JP reranker VH delta | -4.4pp | EVID-20260619-1335-jp-reranker |
| MeCab single-token rate | 100.0% (vs 30.6% default) | EVID-20260619-1335-mecab-dict |

## Current valid scripts / notebooks

- `scripts/compute_all_figures.py` — unified number generator
- `scripts/eval_ablation.py` — ablation evaluation runner
- `scripts/eval_jp_reranker_vh.py` — JP reranker evaluation
- `scripts/build_mecab_materials_dict.py` — MeCab dictionary builder

## Current valid figures / tables

- Table 1: Ablation results (7 conditions) — from paper_data.json
- Table 2: Per-difficulty breakdown — from paper_data.json

## Known exclusions

- All old proposed_result*.csv files (SUPERSEDED by ablation full condition)
- Old paper_figures.json (SUPERSEDED by paper_data.json)
- Old compute_paper_figures.py (SUPERSEDED by compute_all_figures.py)
- Old t2sql_materials_paper.tex/pdf (to be deleted per user request)
- gpt-4o-mini (DO-NOT-USE per user instruction)

## Pure element reference data

- 89 elements from OQMD ground-state calculations [EVID-20260602-0045-oqmd-pure-elements]
- Table: `pure_element_reference` (element_symbol FK → element.symbol)
- Each element: energy_per_atom, volume_per_atom, ground_state_spacegroup, stability, band_gap, n_polymorphs
- Source: OQMD REST API v1 (https://oqmd.org/oqmdapi/formationenergy?filter=ntypes=1)
- 3,212 polymorphs total, lowest delta_e per element selected as ground state
- View: `formation_enthalpy` computes corrected formation enthalpy using pure element references
- DB total: 31 tables, 1559 material entries (1470 compounds + 89 pure elements)
- Schema graph: 31 tables, 204 columns, 70 FK edges

## Remaining uncertainties

- LLM non-determinism: same query may yield different SQL across runs (~1-2pp variance)
- Ablation was single-run per condition (no confidence intervals computed)
- Reranker A/B eval (90q) used a different query subset than ablation (100q)
