# Literature Search Demo
# 文献検索デモ

This document demonstrates 5 example queries against the literature graph,
showing the two-stage search (embedding + structured filter) results,
recommended FeatureSet candidates, and suggested MInt workflow templates.

---

## Query 1: Composition-only yield strength with small dataset

**Query**: `yield_strength regression, composition only, N < 300, HEA`

**Structured Filter**:
- `materials_domain = HEA`
- `task = yield_strength`
- `inputs = composition_only`
- `max_data_size = 300`

**Expected Top-5 Results**:

| Rank | Paper | Model | N | Split | R$^2$ | Key Features |
|------|-------|-------|---|-------|-------|--------------|
| 1 | actamat.2019.03.010 | XGBoost | 200 | random | 0.65 | VEC, delta\_r, dS\_mix, dH\_mix, Tm\_avg |
| 2 | scriptamat.2019.07.039 | Lasso | 180 | random | 0.59 | dS\_mix, VEC, delta\_r, dH\_mix |
| 3 | msea.2020.139038 | GP | 160 | random | 0.60 | delta\_r, VEC, dH\_mix, elastic\_mismatch |
| 4 | jmrt.2020.08.072 | RF | 150 | random | 0.61 | d\_elec\_avg, Vm\_var, VEC, delta\_r, dH\_mix |
| 5 | intermet.2021.107134 | SymbolicRegression | 200 | random | 0.60 | delta\_r, VEC, dS\_mix, Tm\_avg |

**Recommended FeatureSet** (`FS_BASE+LIT_TOP3`):
- Base: r\_avg, delta\_r, dS\_mix, dH\_mix, VEC, delta\_EN, Tm\_avg, mass\_avg
- Added: elastic\_mismatch, d\_elec\_avg, Vm\_var

**Recommended MInt Template**: WF-XGB (most frequent model family in results)

---

## Query 2: Blocked split for robust extrapolation evaluation

**Query**: `yield_strength prediction with blocked or leave-element-out split for HEA`

**Structured Filter**:
- `materials_domain = HEA`
- `task = yield_strength`
- `split_policy = blocked` OR `leave_element_out`

**Expected Top-5 Results**:

| Rank | Paper | Model | N | Split | R$^2$ | Key Features |
|------|-------|-------|---|-------|-------|--------------|
| 1 | actamat.2020.09.068 | Ridge | 190 | blocked | 0.66 | delta\_r, VEC, dH\_mix, ss\_formation |
| 2 | scriptamat.2021.113751 | XGBoost | 280 | blocked | 0.68 | VEC, delta\_r, dS\_mix, omega, elastic\_mismatch |
| 3 | s41524-020-00467-6 | DNN | 280 | leave\_element\_out | 0.68 | VEC, delta\_r, dS\_mix, d\_elec\_avg |
| 4 | commatsci.2020.109871 | XGBoost | 250 | leave\_element\_out | 0.48 | VEC, delta\_r, dS\_mix, dH\_mix, Tm\_avg |
| 5 | msea.2022.142752 | DeepEnsemble | 220 | leave\_element\_out | 0.64 | VEC, delta\_r, dS\_mix, d\_elec\_avg, elastic\_mismatch |

**Recommended FeatureSet** (`FS_BASE+LIT_TOP4`):
- Base: r\_avg, delta\_r, dS\_mix, dH\_mix, VEC, delta\_EN, Tm\_avg, mass\_avg
- Added: ss\_formation, omega, elastic\_mismatch, d\_elec\_avg

**Recommended MInt Template**: WF-ENS (uncertainty quantification critical for extrapolation)

---

## Query 3: Linear models for feature interpretability

**Query**: `interpretable linear model for HEA yield strength, coefficient analysis`

**Structured Filter**:
- `materials_domain = HEA`
- `model_family = linear`

**Expected Top-5 Results**:

| Rank | Paper | Model | N | Split | R$^2$ | Key Features |
|------|-------|-------|---|-------|-------|--------------|
| 1 | actamat.2021.116800 | Ridge | 300 | random | 0.58 | VEC, delta\_r, dS\_mix, dH\_mix, delta\_EN, Tm\_avg, mass\_avg |
| 2 | actamat.2020.09.068 | Ridge | 190 | blocked | 0.66 | delta\_r, VEC, dH\_mix, ss\_formation |
| 3 | scriptamat.2019.07.039 | Lasso | 180 | random | 0.59 | dS\_mix, VEC, delta\_r, dH\_mix |
| 4 | actamat.2020.02.054 | LogisticRegression | 350 | random | - | omega, delta\_r, VEC, dH\_mix |

**Recommended FeatureSet** (`FS_BASE+LIT_TOP2`):
- Base: r\_avg, delta\_r, dS\_mix, dH\_mix, VEC, delta\_EN, Tm\_avg, mass\_avg
- Added: ss\_formation, omega

**Recommended MInt Template**: WF-LIN (coefficient sign analysis and residual diagnostics)

---

## Query 4: Uncertainty-aware prediction for new compositions

**Query**: `uncertainty quantification for HEA mechanical properties, ensemble or GP`

**Structured Filter**:
- `materials_domain = HEA`
- `task = yield_strength`

**Expected Top-5 Results**:

| Rank | Paper | Model | N | Split | R$^2$ | Key Features |
|------|-------|-------|---|-------|-------|--------------|
| 1 | msea.2022.142752 | DeepEnsemble | 220 | leave\_element\_out | 0.64 | VEC, delta\_r, dS\_mix, d\_elec\_avg, elastic\_mismatch |
| 2 | msea.2020.139038 | GP | 160 | random | 0.60 | delta\_r, VEC, dH\_mix, elastic\_mismatch |
| 3 | s41467-019-10533-1 | GP | 100 | leave\_element\_out | 0.55 | VEC, delta\_r, Tm\_avg, dH\_mix |
| 4 | msea.2021.141044 | GP | 60 | random | 0.50 | VEC, delta\_r, dH\_mix, dS\_mix |
| 5 | commatsci.2021.110381 | GP | 180 | random | 0.63 | VEC, delta\_r, dS\_mix, Tm\_avg |

**Recommended FeatureSet** (`FS_BASE+LIT_TOP2`):
- Base: r\_avg, delta\_r, dS\_mix, dH\_mix, VEC, delta\_EN, Tm\_avg, mass\_avg
- Added: elastic\_mismatch, d\_elec\_avg

**Recommended MInt Template**: WF-ENS (seed-varied ensemble for epistemic uncertainty)

---

## Query 5: Best overall performance with all available features

**Query**: `best performing model for HEA yield strength, any features, large dataset`

**Structured Filter**:
- `materials_domain = HEA`
- `task = yield_strength`
- `min_data_size = 200`

**Expected Top-5 Results**:

| Rank | Paper | Model | N | Split | R$^2$ | Key Features |
|------|-------|-------|---|-------|-------|--------------|
| 1 | actamat.2021.116800 | XGBoost | 300 | random | 0.72 | VEC, delta\_r, dS\_mix, dH\_mix, delta\_EN, Tm\_avg, omega |
| 2 | actamat.2022.117431 | GradientBoosting | 250 | random | 0.70 | VEC, delta\_r, dH\_mix, dS\_mix, omega, d\_elec\_avg |
| 3 | scriptamat.2021.113751 | XGBoost | 280 | random | 0.74 | VEC, delta\_r, dS\_mix, omega, elastic\_mismatch |
| 4 | commatsci.2022.111218 | GeneticProgramming | 260 | random | 0.72 | VEC, delta\_r, Tm\_avg, itinerant\_proxy |
| 5 | commatsci.2020.109871 | XGBoost | 250 | leave\_element\_out | 0.48 | VEC, delta\_r, dS\_mix, dH\_mix, Tm\_avg |

**Recommended FeatureSet** (`FS_BASE+LIT_TOP4`):
- Base: r\_avg, delta\_r, dS\_mix, dH\_mix, VEC, delta\_EN, Tm\_avg, mass\_avg
- Added: omega, d\_elec\_avg, elastic\_mismatch, itinerant\_proxy

**Recommended MInt Template**: WF-XGB (tree-based models dominate top results)

---

## Summary

| Query | Focus | Recommended FS | Added Features | MInt Template |
|-------|-------|---------------|---------------|---------------|
| 1 | Small data, composition-only | FS\_BASE+LIT\_TOP3 | elastic\_mismatch, d\_elec\_avg, Vm\_var | WF-XGB |
| 2 | Blocked split, extrapolation | FS\_BASE+LIT\_TOP4 | ss\_formation, omega, elastic\_mismatch, d\_elec\_avg | WF-ENS |
| 3 | Interpretability, linear | FS\_BASE+LIT\_TOP2 | ss\_formation, omega | WF-LIN |
| 4 | Uncertainty quantification | FS\_BASE+LIT\_TOP2 | elastic\_mismatch, d\_elec\_avg | WF-ENS |
| 5 | Best performance, large data | FS\_BASE+LIT\_TOP4 | omega, d\_elec\_avg, elastic\_mismatch, itinerant\_proxy | WF-XGB |

**Key Observations**:
- VEC and delta\_r appear in virtually all top workflows across all queries.
- omega and elastic\_mismatch are the most frequently recommended additions from literature.
- Blocked / leave-element-out splits consistently show lower R$^2$ than random, reinforcing the need for robust evaluation.
- Uncertainty-aware models (GP, ensemble) are strongly recommended for extrapolation tasks.
