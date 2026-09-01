#!/usr/bin/env python3
"""Translate the 100-query ablation dataset questions to English.

Produces evaluation/evaluation_dataset_en.jsonl with identical fields to
evaluation/evaluation_dataset.jsonl except `question` (English) plus
`question_ja` (original). Gold SQL / expected results / difficulty are
unchanged so the pair forms a language-only contrast.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import openai

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "evaluation" / "evaluation_dataset.jsonl"
DST = PROJECT / "evaluation" / "evaluation_dataset_en.jsonl"

SYSTEM = (
    "You translate Japanese natural-language database questions about an "
    "inorganic materials database into English. Preserve the exact meaning, "
    "all numeric conditions, units, element names, phase/prototype names "
    "(e.g. L1\u2082, B2, NaCl, NiAs, BiF\u2083), and materials-science terminology "
    "(e.g. formation energy, energy above hull, bulk modulus, band gap, "
    "space group, lattice constant/parameter). Do not add or drop any "
    "condition, column request, ordering request, or count request. "
    "Return only the English question text."
)


def main() -> None:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("TRANSLATE_MODEL", os.getenv("LLM_MODEL", "gpt-5.5"))
    out = []
    existing: dict[str, dict] = {}
    if DST.exists():
        with open(DST) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[row["id"]] = row
    with open(SRC) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, row in enumerate(rows):
        if row["id"] in existing:
            out.append(existing[row["id"]])
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]},
            ],
        )
        en = (resp.choices[0].message.content or "").strip()
        if not en:
            print(f"EMPTY translation for {row['id']}", file=sys.stderr)
            sys.exit(1)
        new = dict(row)
        new["question_ja"] = row["question"]
        new["question"] = en
        out.append(new)
        print(f"[{i+1}/{len(rows)}] {row['id']}: {en}")
        with open(DST, "w") as g:
            for r in out:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} rows to {DST}")


if __name__ == "__main__":
    main()
