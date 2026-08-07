#!/usr/bin/env python3
"""Generate a per-prototype evaluation dataset from existing gold-sql templates."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT / "evaluation" / "gold_sql"
EXPECTED_DIR = PROJECT / "evaluation" / "expected_results"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

# Query templates derived from existing L1_2 gold SQLs.
TEMPLATES = {
    "easy": {
        "template": "{proto}型構造を持つ化合物を一覧にして。",
        "sql_file": "q_easy_001.sql",
    },
    "medium": {
        "template": "{element}を含む{proto}型化合物を抽出して。",
        "sql_file": "q_easy_002.sql",
    },
    "hard": {
        "template": "安定な{proto}型化合物を形成エネルギーが低い順に出して。",
        "sql_file": "q_medium_005.sql",
    },
    "very_hard": {
        "template": "{proto}型化合物の元素組み合わせと安定性の関係をAサイト・Bサイト別に集計して。",
        "sql_file": "q_vhard_005.sql",
    },
}

PROTOTYPES = {
    "L12": {"element": "Ni"},
    "B2": {"element": "Zr"},
    "NaCl": {"element": "Sc"},
    "NiAs": {"element": "Ti"},
    "BiF3": {"element": "Si"},
}


def adapt_sql(sql: str, proto: str, element: str) -> str:
    """Replace L12 -> proto and Ni -> element in a gold SQL."""
    sql = re.sub(r"'L12'", f"'{proto}'", sql)
    # The OR pattern for prototype/strukturbericht is 'L12' on both sides;
    # after the replacement above both sides already point to the same proto.
    # For prototypes whose strukturbericht differs (NaCl->B1, BiF3->D0_3),
    # we additionally replace the second literal by the actual strukturbericht.
    proto_strukturbericht = {"L12": "L12", "B2": "B2", "NaCl": "B1", "NiAs": "B8_1", "BiF3": "D0_3"}
    # Find pattern: s.prototype = 'X' OR s.strukturbericht = 'X'
    # Replace only the strukturbericht literal if it differs.
    sb = proto_strukturbericht[proto]
    if sb != proto:
        sql = re.sub(r"(s\.strukturbericht\s*=\s*)'[^']+'", rf"\1'{sb}'", sql)
    # Replace element literal Ni (used in medium template)
    sql = re.sub(r"'Ni'", f"'{element}'", sql)
    return sql


def main() -> None:
    conn = psycopg.connect(CONNINFO)
    dataset = []
    for proto, info in PROTOTYPES.items():
        for diff, cfg in TEMPLATES.items():
            qid = f"q_proto_{proto.lower()}_{diff}_001"
            question = cfg["template"].format(proto=proto, element=info["element"])
            sql_path = GOLD_DIR / cfg["sql_file"]
            sql = adapt_sql(sql_path.read_text(encoding="utf-8"), proto, info["element"])
            gold_out = GOLD_DIR / f"{qid}.sql"
            gold_out.write_text(sql, encoding="utf-8")
            # Execute to generate expected results
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
                cur.execute(sql)  # type: ignore[arg-type]
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchall()]
            expected = {"columns": columns, "rows": rows}
            expected_path = EXPECTED_DIR / f"{qid}.json"
            with open(expected_path, "w", encoding="utf-8") as f:
                json.dump(expected, f, ensure_ascii=False, indent=2)
            dataset.append({
                "id": qid,
                "question": question,
                "difficulty": diff,
                "gold_sql_path": f"gold_sql/{qid}.sql",
                "expected_result_path": f"expected_results/{qid}.json",
            })
    dataset_path = PROJECT / "evaluation" / "prototype_evaluation_dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(dataset)} prototype queries -> {dataset_path}")
    print("Per-query row counts:")
    for item in dataset:
        qid = item["id"]
        expected = json.loads((EXPECTED_DIR / f"{qid}.json").read_text())
        print(f"  {qid}: {len(expected['rows'])} rows")
    conn.close()


if __name__ == "__main__":
    main()
