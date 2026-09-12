"""Build the canonical catalog of all unique evaluation queries.

Reads the canonical corpora (main 245, transfer 20, obfuscated transfer
20, MP transfer 15 — 300 unique queries in total), parses each gold SQL
with sqlglot, and emits:

- evaluation/query_catalog.csv  (one row per query, machine-readable)
- evaluation/query_catalog.json (same content plus summary matrices)

The build asserts that qids are unique and that the catalog covers
exactly the canonical gold-SQL corpus (gold_sql/ + gold_sql_obfuscated/
+ gold_sql_mp/).  Per-experiment run datasets (evaluation_dataset.jsonl,
expert_evaluation_dataset.jsonl, cte15_dataset.jsonl, ...) overlap with
the main corpus and are intentionally NOT concatenated here.

Each query gets: id, question, eval set, category, difficulty (the
``difficulty_cte_folded`` column keeps the original design-time label and
only folds CTE queries into very_hard; it is unrelated to the post-hoc
structural-complexity score of ``compute_unified_difficulty.py``), CTE
flag, tables used, table count, join count, aggregation flag, subquery
flag, and SQL feature tags.
"""

import csv
import json
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"

DATASETS = [
    ("main", "main_evaluation_dataset.jsonl"),
    ("transfer", "transfer_evaluation_dataset.jsonl"),
    ("transfer_obfuscated", "transfer_obfuscated_evaluation_dataset.jsonl"),
    ("mp_transfer", "mp_transfer_evaluation_dataset.jsonl"),
]

CATEGORY_TABLES = {
    "surface_gb_defect": {
        "surface_energy", "grain_boundary", "material_defect",
    },
    "literature_application": {
        "material_reference", "material_application", "material_synthesis",
        "experimental_measurement", "measured_property",
    },
    "electronic_magnetic_thermal": {
        "band_structure", "density_of_states", "magnetic_property",
        "thermal_property", "elastic_tensor", "calculated_property",
        "property_definition",
    },
    "stability": {
        "phase_stability", "phase_diagram_entry", "formation_enthalpy",
        "pure_element_reference",
    },
    "structure": {
        "structure", "prototype_definition", "space_group",
    },
    "composition": {
        "composition", "element", "material_alloy_system",
        "element_property",
    },
}
CATEGORY_ORDER = [
    "surface_gb_defect", "literature_application",
    "electronic_magnetic_thermal", "stability", "structure", "composition",
]
CATEGORY_JA = {
    "multistep": "多段計算",
    "transfer": "転用（OQMD）",
    "transfer_obfuscated": "転用（難読化）",
    "mp_transfer": "転用（MP）",
    "surface_gb_defect": "表面・粒界・欠陥",
    "literature_application": "文献・応用",
    "electronic_magnetic_thermal": "電子構造・磁気・熱",
    "stability": "安定性",
    "structure": "構造",
    "composition": "組成",
}


def parse_sql_features(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="postgres")
    tables = sorted({
        t.name.lower() for t in tree.find_all(exp.Table)
    })
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    base_tables = sorted(set(tables) - cte_names)
    n_joins = len(list(tree.find_all(exp.Join)))
    has_cte = bool(cte_names)
    has_window = bool(list(tree.find_all(exp.Window)))
    agg_funcs = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
    has_agg = any(bool(list(tree.find_all(f))) for f in agg_funcs)
    selects = list(tree.find_all(exp.Select))
    n_sub_selects = len(selects) - 1
    has_group_by = bool(list(tree.find_all(exp.Group)))
    has_having = bool(list(tree.find_all(exp.Having)))
    has_distinct = bool(list(tree.find_all(exp.Distinct)))
    has_case = bool(list(tree.find_all(exp.Case)))
    features = []
    if has_cte:
        features.append("CTE")
    if has_window:
        features.append("window")
    if has_agg:
        features.append("aggregation")
    if has_group_by:
        features.append("group_by")
    if has_having:
        features.append("having")
    if n_sub_selects > (len(cte_names) if has_cte else 0):
        features.append("subquery")
    if has_distinct:
        features.append("distinct")
    if has_case:
        features.append("case")
    return {
        "tables": base_tables,
        "n_tables": len(base_tables),
        "n_joins": n_joins,
        "has_cte": has_cte,
        "has_aggregation": has_agg,
        "has_subquery": "subquery" in features,
        "sql_features": features,
    }


def categorize(tables: list[str], has_cte: bool) -> str:
    if has_cte or "pure_element_reference" in tables or "formation_enthalpy" in tables:
        return "multistep"
    tset = set(tables)
    for cat in CATEGORY_ORDER:
        if tset & CATEGORY_TABLES[cat]:
            return cat
    return "composition"


def load_gold_sql(rec: dict) -> str:
    if "gold_sql" in rec:
        return rec["gold_sql"]
    return (EVAL / rec["gold_sql_path"]).read_text()


def expected_result_path(eval_set: str, rec: dict) -> Path | None:
    if rec.get("expected_result_path"):
        return EVAL / rec["expected_result_path"]
    if eval_set == "mp_transfer":
        return EVAL / "expected_results_mp_transfer" / f"{rec['id']}.json"
    return None


def canonical_gold_qids() -> set[str]:
    qids: set[str] = set()
    for d in ("gold_sql", "gold_sql_obfuscated", "gold_sql_mp"):
        qids |= {p.stem for p in (EVAL / d).glob("*.sql")}
    return qids


def main() -> None:
    rows = []
    for eval_set, fname in DATASETS:
        for line in (EVAL / fname).read_text().splitlines():
            rec = json.loads(line)
            sql = load_gold_sql(rec)
            feats = parse_sql_features(sql)
            is_cte_set = eval_set in ("cte15", "cte_pattern")
            difficulty = rec["difficulty"]
            cte_folded_difficulty = "very_hard" if (is_cte_set or feats["has_cte"]) else difficulty
            if eval_set == "main":
                cat = categorize(feats["tables"], feats["has_cte"])
            else:
                cat = eval_set
            rows.append({
                "qid": rec["id"],
                "question": rec["question"],
                "eval_set": eval_set,
                "category": cat,
                "category_ja": CATEGORY_JA[cat],
                "difficulty_original": difficulty,
                "difficulty_cte_folded": cte_folded_difficulty,
                "is_cte": feats["has_cte"] or is_cte_set,
                "tables": ";".join(feats["tables"]),
                "n_tables": feats["n_tables"],
                "n_joins": feats["n_joins"],
                "has_aggregation": feats["has_aggregation"],
                "has_subquery": feats["has_subquery"],
                "sql_features": ";".join(feats["sql_features"]),
                "gold_sql_available": True,
                "expected_result_available": bool(
                    (erp := expected_result_path(eval_set, rec))
                    and erp.exists()
                ),
            })

    qids = [r["qid"] for r in rows]
    assert len(qids) == len(set(qids)), "duplicate qids in catalog"
    gold_qids = canonical_gold_qids()
    assert set(qids) == gold_qids, (
        f"catalog/gold mismatch: missing={sorted(gold_qids - set(qids))} "
        f"extra={sorted(set(qids) - gold_qids)}")
    assert len(rows) == 300, f"expected 300 canonical queries, got {len(rows)}"

    csv_path = EVAL / "query_catalog.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matrix: dict[str, dict[str, int]] = {}
    for r in rows:
        matrix.setdefault(r["category_ja"], {}).setdefault(r["difficulty_cte_folded"], 0)
        matrix[r["category_ja"]][r["difficulty_cte_folded"]] += 1
    by_set: dict[str, int] = {}
    for r in rows:
        by_set[r["eval_set"]] = by_set.get(r["eval_set"], 0) + 1
    feature_counts: dict[str, int] = {}
    for r in rows:
        for feat in r["sql_features"].split(";"):
            if feat:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
    representatives = {}
    for r in rows:
        representatives.setdefault(r["category_ja"], r["question"])

    json_path = EVAL / "query_catalog.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_total": len(rows),
            "by_eval_set": by_set,
            "category_x_difficulty": matrix,
            "sql_feature_counts": feature_counts,
            "representative_questions": representatives,
            "queries": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"total queries: {len(rows)}")
    print(f"by set: {by_set}")
    print("category x difficulty:")
    for cat, d in matrix.items():
        print(f"  {cat}: {d}")
    print(f"features: {feature_counts}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
