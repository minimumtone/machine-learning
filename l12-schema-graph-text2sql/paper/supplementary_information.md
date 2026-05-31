# Supplementary Information

Schema-Graph-Constrained Text-to-SQL for Normalized Relational Materials Databases

---

## S1. Database Schema Definition

```sql
CREATE TABLE material_entry (
    entry_id TEXT PRIMARY KEY,
    source_db TEXT,
    source_material_id TEXT,
    formula TEXT NOT NULL,
    reduced_formula TEXT,
    chemical_system TEXT,
    number_of_elements INTEGER
);

CREATE TABLE composition (
    composition_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    element TEXT NOT NULL,
    atomic_fraction DOUBLE PRECISION NOT NULL,
    site_label TEXT
);

CREATE TABLE structure (
    structure_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    prototype TEXT,
    strukturbericht TEXT,
    formula_type TEXT,
    space_group_number INTEGER,
    crystal_system TEXT,
    lattice_a DOUBLE PRECISION,
    lattice_b DOUBLE PRECISION,
    lattice_c DOUBLE PRECISION,
    volume_per_atom DOUBLE PRECISION,
    space_group TEXT
);

CREATE TABLE calculation (
    calculation_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    method TEXT,
    functional TEXT,
    calculation_type TEXT
);

CREATE TABLE calculated_property (
    property_id TEXT PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    property_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT
);

CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    formation_energy_per_atom DOUBLE PRECISION,
    energy_above_hull DOUBLE PRECISION,
    is_stable BOOLEAN,
    band_gap DOUBLE PRECISION
);

CREATE TABLE prototype_definition (
    prototype_id TEXT PRIMARY KEY,
    prototype_name TEXT,
    strukturbericht TEXT,
    formula_type TEXT,
    description TEXT
);
```

### Indexes

```sql
CREATE INDEX idx_composition_entry_id ON composition(entry_id);
CREATE INDEX idx_composition_element ON composition(element);
CREATE INDEX idx_structure_entry_id ON structure(entry_id);
CREATE INDEX idx_structure_prototype ON structure(prototype);
CREATE INDEX idx_structure_strukturbericht ON structure(strukturbericht);
CREATE INDEX idx_phase_stability_entry_id ON phase_stability(entry_id);
CREATE INDEX idx_phase_energy_above_hull ON phase_stability(energy_above_hull);
CREATE INDEX idx_calculation_entry_id ON calculation(entry_id);
CREATE INDEX idx_property_calculation_id ON calculated_property(calculation_id);
CREATE INDEX idx_property_name ON calculated_property(property_name);
```

---

## S2. Allowed Schema Configuration (allowed_schema.yaml)

```yaml
allowed_tables:
  - material_entry
  - composition
  - structure
  - calculation
  - calculated_property
  - phase_stability
  - prototype_definition

allowed_joins:
  - source_table: composition
    source_column: entry_id
    target_table: material_entry
    target_column: entry_id
  - source_table: structure
    source_column: entry_id
    target_table: material_entry
    target_column: entry_id
  - source_table: calculation
    source_column: entry_id
    target_table: material_entry
    target_column: entry_id
  - source_table: calculated_property
    source_column: calculation_id
    target_table: calculation
    target_column: calculation_id
  - source_table: phase_stability
    source_column: entry_id
    target_table: material_entry
    target_column: entry_id
```

---

## S3. Material Domain Dictionary (material_terms.yaml)

```yaml
prototype_aliases:
  L12:
    - L12, L1₂, L1_2, Cu3Au, Cu₃Au型, ordered FCC, 規則化FCC, gamma prime, γ', ガンマプライム
  B2:
    - B2, CsCl, CsCl型, ordered BCC, 規則化BCC

stability_terms:
  stable:
    condition: phase_stability.energy_above_hull <= 0.001 eV/atom
  metastable:
    condition: phase_stability.energy_above_hull <= 0.05 eV/atom

property_terms:
  formation_energy:
    aliases: [formation energy, 形成エネルギー, 生成エネルギー]
    column: phase_stability.formation_energy_per_atom
  lattice_constant:
    aliases: [lattice constant, 格子定数, lattice_a]
    column: structure.lattice_a
  bulk_modulus:
    aliases: [bulk modulus, 体積弾性率, バルクモジュラス]
    property_name: bulk_modulus (EAV in calculated_property)
  shear_modulus:
    aliases: [shear modulus, せん断弾性率, シアモジュラス]
    property_name: shear_modulus
  energy_above_hull:
    aliases: [energy above hull, ハル上エネルギー, Ehull]
    column: phase_stability.energy_above_hull
  band_gap:
    aliases: [band gap, バンドギャップ, bandgap]
    column: phase_stability.band_gap

elements:
  14 elements registered with Japanese aliases:
  Ni (ニッケル), Al (アルミニウム), Co (コバルト), Ti (チタン),
  Ta (タンタル), Nb (ニオブ), W (タングステン), Fe (鉄),
  Cu (銅), Pt (プラチナ), Ir (イリジウム), Sc (スカンジウム),
  Ga (ガリウム), Ge (ゲルマニウム)
```

---

## S4. LLM Prompt Template (sql_generation_prompt.md)

```
You are a Text-to-SQL generator for a materials database.
Generate only one PostgreSQL SELECT query.

Rules:
- Use only the provided tables.
- Use only the provided columns.
- Use only the provided JOIN clauses.
- Do not invent tables.
- Do not invent columns.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
- Return SQL only.
- Always include a LIMIT clause (default LIMIT 100).

Table aliases:
- material_entry -> m
- composition -> c
- structure -> s
- calculation -> calc
- calculated_property -> cp
- phase_stability -> ps
- prototype_definition -> pd

Material term mappings:
- L1₂, L12, Cu3Au-type, γ' -> structure.prototype = 'L12' OR structure.strukturbericht = 'L12'
- stable -> phase_stability.energy_above_hull <= 0.001
- metastable -> phase_stability.energy_above_hull <= 0.05
- formation energy, 形成エネルギー -> phase_stability.formation_energy_per_atom
- lattice constant, 格子定数 -> structure.lattice_a
- bulk modulus, 体積弾性率 -> calculated_property.property_name = 'bulk_modulus'

Allowed tables:
{allowed_tables}

Allowed columns:
{allowed_columns}

Allowed JOINs:
{allowed_joins}

User query:
{user_query}

SQL:
```

---

## S5. LLM Experiment Conditions

### S5.1. Model and API Settings

| Parameter | Value |
| --- | --- |
| Model identifier | `gpt-5.5` |
| API provider | OpenAI |
| API endpoint | `https://api.openai.com/v1/chat/completions` |
| API date | 2026-05-30 |
| Temperature | N/A (gpt-5.5 does not accept temperature parameter) |
| top_p | N/A (default) |
| max_completion_tokens | 4096 |
| seed | Not supported by gpt-5.5; non-deterministic output |
| Retry policy | No retry; single API call per query per condition |
| Few-shot retrieval | TF-IDF cosine similarity, top_k=3 |
| Schema constraints | Injected into user prompt (tables, columns, types, FK relationships in YAML) |

**Note:** GPT-5 / o-series models use `max_completion_tokens` instead of `max_tokens`, and `temperature` is omitted. The `seed` parameter is not supported, which means outputs are non-deterministic. This is quantified in the reproducibility test (20 queries × 5 runs, SQL exact-match rate 30%, execution success rate 100%).

### S5.2. System Prompt (Full Text)

```
You are a PostgreSQL expert for materials databases. Generate ONLY a single SELECT SQL query.
Rules:
- Use ONLY the tables and columns listed in the schema below
- Always include LIMIT 100 unless the user specifies otherwise
- Use proper JOIN conditions based on foreign key relationships
- For element-based queries, use EXISTS subqueries with the composition table
- Return ONLY the SQL query, no explanation
```

### S5.3. Schema Prompt Structure

The schema prompt injected into the user message contains:
1. Table definitions (name, columns with types)
2. Foreign key relationships (parent → child)
3. Available JOIN paths (pre-computed by Schema Graph)
4. Sample values for key columns (prototypes, stability thresholds)

### S5.4. Reproducibility Test Log Summary

| Metric | Value |
| --- | --- |
| Queries tested | 20 (representative subset) |
| Runs per query | 5 |
| SQL exact-match rate | 30% (6/20 queries produced identical SQL across all 5 runs) |
| Execution success rate | 100% (all 100 executions returned valid results) |
| Result-set consistency | 95% (19/20 queries returned identical result sets across runs) |
| Median latency | 2.8 s |

The low SQL exact-match rate reflects gpt-5.5's tendency to generate syntactically diverse but semantically equivalent SQL (e.g., different column ordering, alias choices, JOIN syntax). The high execution success and result-set consistency rates confirm that this diversity does not affect correctness.

---

## S6. SQL Guard Configuration

### Validation Pipeline

| Check | Action on Failure |
| --- | --- |
| Multiple statement detection | Reject (rejected_security) |
| Forbidden keyword detection (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, COPY) | Reject (rejected_security) |
| SELECT-only constraint | Reject (rejected_syntax) |
| Table whitelist (7 allowed tables) | Reject (rejected_schema) |
| Column whitelist (alias.column pattern) | Reject (rejected_schema) |
| JOIN FK validation (must match allowed_schema.yaml) | Warn (rejected_join) |
| Disallowed function detection (pg_sleep, dblink, lo_import, lo_export, pg_read_file, pg_ls_dir, pg_stat_file) | Reject (rejected_security) |
| Subquery depth limit (max 3) | Reject (rejected_complexity) |
| LIMIT enforcement (default 100) | Auto-modify (modified) |

### Classification Categories

| Classification | Description |
| --- | --- |
| accepted | SQL passes all checks |
| modified | LIMIT added automatically |
| rejected_syntax | Not a SELECT statement |
| rejected_security | Forbidden keyword or function |
| rejected_schema | Disallowed table or column |
| rejected_join | JOIN not matching FK relationship |
| rejected_complexity | Subquery depth exceeds limit |
| clarification_required | Intent unclear, user confirmation needed |

---

## S7. Coverage Score Algorithm

### Purpose

Prevents silent constraint dropping in rule-based mode by quantifying how well the system understands a user query.

### Algorithm

```
Input: user_query (natural language string)
Output: {recognized_constraints, unrecognized_terms, unknown_elements,
         coverage_score, action}

1. Tokenize query (split on whitespace and Japanese particles)
2. For each token:
   a. Check against prototype_aliases → recognized
   b. Check against stability_terms → recognized
   c. Check against property_terms → recognized
   d. Check against elements dictionary → recognized
   e. Check if element-like (regex: [A-Z][a-z]?) and in _ALL_ELEMENTS
      but NOT in known_elements → mark as unknown_element
   f. Otherwise → unrecognized
3. coverage_score = recognized_count / (recognized_count + unrecognized_count)
4. Fallback policy:
   - score >= 0.8 → action = "execute_rule_based"
   - 0.5 <= score < 0.8 → action = "fallback_to_llm"
   - score < 0.5 → action = "clarification_required"
   - If unknown_elements detected → action = "fallback_to_llm" (override)
```

### Example Outputs

| Query | Coverage | Unknown Elements | Action |
| --- | --- | --- | --- |
| Feを含むB2化合物を出して | 0.85 | [] | execute_rule_based |
| Xeを含むB2化合物を出して | 0.55 | [Xe] | fallback_to_llm |
| 今日の天気を教えて | 0.0 | [] | clarification_required |
| band gap > 1.0 eVのB2化合物 | 0.88 | [] | execute_rule_based |

---

## S8. Numeric Condition Parser

### Supported Patterns

| Natural Language Pattern | SQL Condition |
| --- | --- |
| `band gap > 1.0 eV` | `phase_stability.band_gap > 1.0` |
| `band gapが1 eV以上` | `phase_stability.band_gap >= 1.0` |
| `Ehull < 0.05 eV/atom` | `phase_stability.energy_above_hull < 0.05` |
| `形成エネルギーが負` | `phase_stability.formation_energy_per_atom < 0` |
| `格子定数が3.5 Å以上` | `structure.lattice_a >= 3.5` |
| `格子定数が3から4` | `structure.lattice_a BETWEEN 3.0 AND 4.0` |

### Unit Conversion

| Input Unit | Canonical Unit | Conversion Factor |
| --- | --- | --- |
| eV | eV | 1.0 |
| meV/atom | eV/atom | 0.001 |
| Å | Å | 1.0 |
| Angstrom | Å | 1.0 |
| nm | Å | 10.0 |

### Property Column Mapping

| Alias | Column |
| --- | --- |
| band_gap, band gap, バンドギャップ, bandgap | phase_stability.band_gap |
| formation_energy, formation energy, 形成エネルギー, 生成エネルギー | phase_stability.formation_energy_per_atom |
| energy_above_hull, Ehull, ハル上エネルギー | phase_stability.energy_above_hull |
| lattice_a, lattice constant, 格子定数 | structure.lattice_a |
| volume_per_atom, 原子あたり体積, volume | structure.volume_per_atom |

---

## S9. Chemical Formula Parser

### Interpretation Rules

| Input | Interpretation | Query Mode |
| --- | --- | --- |
| NiAl | contains_elements [Ni, Al] | EXISTS subquery for each element |
| Ni3Al | exact_formula (Ni:3, Al:1) | WHERE formula = 'Ni3Al' OR reduced_formula = 'Ni3Al' |
| AlNi3 | exact_formula (Al:1, Ni:3) | WHERE formula = 'AlNi3' |
| FeCoNi | contains_elements [Fe, Co, Ni] | EXISTS subquery ×3 |

### Disambiguation Priority

1. If token contains digits → exact_formula (e.g., Ni3Al)
2. If token is 2 capital letters → contains_elements (e.g., NiAl)
3. If token is 3+ elements → contains_elements (e.g., FeCoNi)
4. If ambiguous → prefer contains_elements, flag for LLM clarification

---

## S10. Baseline Comparison Results (7 Conditions, 57 Queries, gpt-5.5)

### S10.1. Full Metrics Table

| Condition | Exec Success | Result Correctness | Hallucinated Schema | Silent Drop | Unnecessary JOINs | Avg Rows |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Naive rule-based | 96.5% (55/57) | 79.3% (45/57) | 0 | 12.1% (7/57) | 3 | 54.4 |
| 2. LLM-only (no schema info) | 1.8% (1/57) | 1.8% (1/57) | 56 | 0% | N/A | N/A |
| 3. LLM + schema prompt | 100.0% (57/57) | 100.0% (57/57) | 0 | 0% | 2 | 30.2 |
| 4. LLM + schema + few-shot | 100.0% (57/57) | 100.0% (57/57) | 0 | 0% | 1 | 43.6 |
| 5. Schema Graph + Rule-based | 100.0% (57/57) | 100.0% (57/57) | 0 | 0% | 0 | 44.9 |
| 6. Schema Graph + LLM (no RAG) | 100.0% (57/57) | 100.0% (57/57) | 0 | 0% | 0 | 44.1 |
| 7. Schema Graph + LLM + RAG | 100.0% (57/57) | 100.0% (57/57) | 0 | 0% | 0 | 44.7 |

**Metric definitions:**
- **Exec Success**: SQL executes without error
- **Result Correctness**: Returned result set matches expected results (human-verified)
- **Hallucinated Schema**: SQL references columns/tables not in the database schema
- **Silent Drop**: Query constraints silently ignored (e.g., unrecognized element dropped)
- **Unnecessary JOINs**: JOINs not required by the query but included by the generator

### Key Comparison: Multi-element AND Queries

| Query | Naive RB (rows) | Schema Graph RB (rows) | Note |
| --- | --- | --- | --- |
| NiとAlを両方含む化合物 | 97 (incorrect) | 4 (correct) | Naive uses row-level AND; SG uses EXISTS |
| FeとAlを含むB2化合物 | 55 (incorrect) | 4 (correct) | Same pattern |
| NiとCoを含むL12化合物 | 12 (incorrect) | 0 (correct) | No NiCo L12 in DB |

---

## S11. LLM Reproducibility Test (20 Queries × 5 Runs, gpt-5.5)

| Metric | Value |
| --- | --- |
| Model | gpt-5.5 |
| Temperature | N/A (not supported) |
| max_completion_tokens | 4096 |
| SQL consistency rate | 30.0% (6/20 queries produced identical SQL across 5 runs) |
| Execution success rate | 100.0% (100/100 runs) |
| Latency: average | 3,609 ms |
| Latency: median | 3,194 ms |

**Note:** gpt-5.5 produces more varied but functionally equivalent SQL. All 100 runs executed successfully. The lower SQL consistency rate reflects gpt-5.5's tendency to generate structurally different but semantically correct queries (e.g., different column ordering, JOIN order, aliasing).

---

## S12. VASP-Forum-Inspired Stress Test Results (100 Queries, gpt-5.5)

### S12.1. Intent Classifier Before/After Comparison

| Category | Count | LLM-only Correct | LLM-only Acc. | +IntentClassifier Correct | +IC Acc. |
| --- | --- | --- | --- | --- | --- |
| SQL-answerable | 22 | 22 | 100.0% | 22 | 100.0% |
| SQL-answerable-numeric | 21 | 21 | 100.0% | 21 | 100.0% |
| ambiguous | 25 | 10 | 40.0% | 10 | 40.0% |
| out-of-scope | 22 | 0 | **0.0%** | 22 | **100.0%** |
| unsafe | 10 | 2 | **20.0%** | 10 | **100.0%** |
| **Total** | **100** | **55** | **55.0%** | **85** | **85.0%** |

### S12.2. Key Metrics (After Intent Classifier)

| Metric | LLM-only | +Intent Classifier |
| --- | --- | --- |
| Overall accuracy | 55.0% (55/100) | 85.0% (85/100) |
| SQL-answerable accuracy | 100.0% (43/43) | 100.0% (43/43) |
| Out-of-scope rejection rate | 0.0% (0/22) | 100.0% (22/22) |
| Unsafe rejection rate | 20.0% (2/10) | 100.0% (10/10) |
| Silent constraint drops | 0 | 0 |
| Hallucinated schema | 0 | 0 |

### S12.3. Failure Mode Analysis (After Intent Classifier)

| Failure Mode | LLM-only | +Intent Classifier | Description |
| --- | --- | --- | --- |
| should_have_clarified | 13 | 13 | Ambiguous queries: system generated SQL instead of asking clarification |
| generated_sql_for_out_of_scope | 22 | 0 | VASP workflow questions: eliminated by Intent Classifier |
| unsafe_sql_executed | 5 | 0 | Unsafe input: eliminated by Intent Classifier + SQL Guard |

### Known Limitations Revealed

1. **Out-of-scope detection**: The LLM mode generates SQL for VASP workflow questions (Q071-Q090) because the LLM interprets any materials-related text as a potential database query. A query classifier or intent detector is needed.
2. **Ambiguity handling**: The system rarely asks for clarification (1/25 ambiguous queries). Threshold-based heuristics or explicit ambiguity detection should be added.
3. **Unsafe input filtering**: While SQL injection (DROP, UPDATE) is blocked by the SQL Guard, some adversarial inputs (contradictory conditions, extreme LIMIT requests) are not caught before LLM generation.

---

## S12b. RAG Ablation Results (4 Conditions, 50 Queries, gpt-5.5)

| Condition | SQL Exec Success | Description |
| --- | --- | --- |
| 1. No examples (schema only) | 100.0% (50/50) | Constrained prompt with no few-shot examples |
| 2. Manual examples only | 100.0% (50/50) | Only manually curated seed examples |
| 3. Paper-extracted examples only | 100.0% (50/50) | Only examples extracted from LaTeX paper |
| 4. All examples (full RAG) | 100.0% (50/50) | All sources combined via TF-IDF retrieval |

**Interpretation:** At this task difficulty level and with gpt-5.5, the schema-constrained prompt alone achieves perfect execution success. RAG few-shot examples provide no measurable improvement. This ceiling effect suggests that the schema graph constraint is the primary contributor to SQL quality, not the few-shot examples. A more challenging query set (multi-table aggregations, window functions, etc.) would be needed to differentiate RAG conditions.

---

## S13. Failure Mode Transition Table

| Failure Mode | Cause | Previous Behavior | Revised Behavior |
| --- | --- | --- | --- |
| Silent constraint dropping | Unknown element (e.g., Xe) | Over-broad result (all B2) | Fallback to LLM or clarification |
| Formula ambiguity | NiAl vs Ni+Al | Inconsistent interpretation | Candidate interpretations + priority |
| Numeric condition failure | No parser | Ignored or LLM-only | Regex numeric parser with unit conversion |
| Invalid LLM SQL | Hallucinated column | Guard rejection | Column whitelist + alias resolution |
| Multi-element AND error | Row-level conjunction | Empty or over-broad result | EXISTS subquery pattern |

---

## S14. Curated Regression Test Summary (60 Tests)

All 60 tests pass. These are development safety checks, not ground-truth evaluation.

- 39 original regression tests (element, prototype, stability, sorting, adversarial)
- 6 SQL validator tests (imaginary column, DROP rejection, multi-statement, unknown table)
- 15 coverage score and parser tests (numeric conditions, chemical formula parsing, coverage scoring)

---

## S15. Scalability Measurement (Rule-based vs LLM, 909 entries, 7 tables)

### Rule-based Latency by Query Type

| Query Type | Avg Total (ms) | Extract (ms) | Generate (ms) | Execute (ms) |
| --- | --- | --- | --- | --- |
| simple_element | 60 | 35 | 10 | 9 |
| simple_prototype | 35 | 10 | 10 | 8 |
| numeric_condition | 35 | 10 | 10 | 9 |
| multi_element | 34 | 9 | 9 | 9 |
| sorting_limit | 54 | 10 | 19 | 9 |
| compound_query | 43 | 10 | 13 | 9 |

### LLM (gpt-5.5) Latency by Query Type

| Query Type | Avg Total (ms) | Generate/LLM (ms) |
| --- | --- | --- |
| simple_element | 2,809 | 2,785 |
| simple_prototype | 3,842 | 3,817 |
| numeric_condition | 5,687 | 5,661 |
| multi_element | 6,085 | 6,056 |
| sorting_limit | 2,635 | 2,603 |
| compound_query | 3,179 | 3,150 |

### Rule-based vs LLM Speed Comparison

| Query | RB (ms) | LLM (ms) | Speedup |
| --- | --- | --- | --- |
| Feを含むB2化合物を出して | 32 | 2,861 | 89x |
| band gap > 1.0 eVのB2化合物を出して | 32 | 3,072 | 97x |
| NiとAlを両方含む化合物を出して | 31 | 5,949 | 194x |
| 安定なL1₂化合物を形成エネルギーが低い順に出して | 49 | 2,371 | 49x |

**Key finding:** Rule-based generation is 49-194x faster than LLM. The LLM latency is dominated by API call time (~99% of total). Local components (extract, link, validate, execute) contribute <50ms total. For queries within the registered vocabulary, rule-based mode provides both correct results and sub-100ms response times.

---

## S16. Intent Classifier

The intent classifier is a pre-LLM gate that rejects out-of-scope queries before expensive API calls.

### Classification Categories

| Intent | Action | Example |
| --- | --- | --- |
| db_query | Pass to SQL pipeline | "Feを含むB2化合物を出して" |
| out_of_scope | Reject with explanation | "VASPでmBJ+SOCを使うときのINCAR設定を教えて" |
| unsafe | Reject immediately | "DROP TABLE material_entry;" |
| greeting | Reject politely | "こんにちは" |
| ambiguous | Request clarification | "B2" (too short) |

### Pattern Matching Approach

- 20 VASP/DFT workflow patterns (INCAR, KPOINTS, ENCUT, convergence, etc.)
- 9 DB query patterns (B2/L12 compounds, formation energy, stability, etc.)
- 3 greeting patterns
- 3 unsafe patterns (DROP, DELETE, UPDATE, secrets)
- 6 out-of-schema property patterns (shear modulus, magnetic moment, etc.)

When both VASP and DB patterns match, the classifier uses score comparison: VASP score ≥ DB score → out_of_scope.

---

## S17. Data Availability

The source code, schema definitions, query test sets, and evaluation scripts will be made available in a public repository upon publication. The OQMD-derived data can be regenerated using the provided data acquisition scripts from the public OQMD API (https://oqmd.org/api/).

### Reproducibility Checklist

| Item | Available |
| --- | --- |
| Database schema SQL | Yes (db/schema.sql) |
| Data acquisition script | Yes (db/oqmd_loader.py) |
| Material terms dictionary | Yes (llm/material_terms.yaml) |
| Allowed schema config | Yes (safety/allowed_schema.yaml) |
| LLM prompt template | Yes (llm/prompt_templates/) |
| SQL Guard implementation | Yes (safety/sql_validator.py) |
| Entity extractor + parsers | Yes (llm/entity_extractor.py) |
| Schema Graph builder | Yes (graph/graph_builder.py) |
| Test query sets | Yes (experiments/) |
| Evaluation scripts | Yes (experiments/run_*.py) |
| Few-shot example store | Yes (llm/few_shot_store.py) |
| Intent classifier | Yes (llm/intent_classifier.py) |
| VASP stress test (100 queries) | Yes (experiments/run_vasp_stress_test.py) |
| Scalability benchmark | Yes (experiments/run_scalability.py) |
| RAG ablation script | Yes (experiments/run_all_experiments.py) |

### Items NOT publicly available

| Item | Reason |
| --- | --- |
| OpenAI API key | Credential |
| Raw LLM API logs | Contains API metadata |
| Internal database connection strings | Infrastructure credential |
