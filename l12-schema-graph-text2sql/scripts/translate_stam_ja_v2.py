#!/usr/bin/env python3
"""Translate stam-m.tex into Japanese, preserving LaTeX commands/structure."""
import os
import re
import sys
import hashlib
import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"
CACHE_DIR = "/tmp/stam_ja_cache_v2"

PROMPT = """You are translating an English LaTeX manuscript into Japanese.
Instructions:
- Translate only natural English prose into natural Japanese academic text.
- Keep all LaTeX commands, environments, cite keys, labels, refs, math ($...$ and equations), numbers, units, and chemical formulas unchanged.
- Translate section/subsection titles and figure/table captions to Japanese, but leave table body rows, column headers, and code unchanged.
- Preserve itemize/enumerate structure; translate item text only.
- Do not add any explanatory text before or after the code block.
- Output only the translated LaTeX (no ```)."""


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def translate_chunk(text: str) -> str:
    if not text.strip():
        return text
    h = chunk_hash(text)
    cache_path = os.path.join(CACHE_DIR, f"{h}.tex")
    if os.path.exists(cache_path):
        return open(cache_path, "r", encoding="utf-8").read()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=8192,
    )
    result = resp.choices[0].message.content
    os.makedirs(CACHE_DIR, exist_ok=True)
    open(cache_path, "w", encoding="utf-8").write(result)
    return result


def split_recursively(text: str, max_chars: int = 8000) -> list[str]:
    """Split LaTeX into sizable, semantically meaningful chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # Try major structural splits
    for pattern in [
        r"\n(?=\\subsection\{)",
        r"\n(?=\\subsubsection\{)",
        r"\n(?=\\paragraph\{)",
        r"\n\s*\n",
    ]:
        parts = re.split(pattern, text)
        if len(parts) > 1:
            out = []
            cur = ""
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(cur) + len(p) + 2 <= max_chars:
                    cur = (cur + "\n\n" + p).strip() if cur else p
                else:
                    if cur:
                        out.extend(split_recursively(cur, max_chars))
                    cur = p
            if cur:
                out.extend(split_recursively(cur, max_chars))
            return out
    # Fallback: hard split
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


def top_level_chunks(text: str) -> list[str]:
    """Split by top-level LaTeX structural commands, keeping each header with its body."""
    # Pattern before structural markers, but include the marker in the chunk.
    markers = [
        r"\\begin\{abstract\}\n",
        r"\\end\{abstract\}\n?",
        r"\\begin\{keywords\}\n",
        r"\\end\{keywords\}\n?",
        r"\\section\{[^}]+\}\n?",
        r"\\appendix\n",
        r"\\bibliographystyle\{[^}]+\}",
        r"\\end\{document\}",
    ]
    pattern = "(" + "|".join(markers) + ")"
    parts = re.split(f"(?={pattern})", text)
    merged = []
    cur = ""
    for p in parts:
        if not p:
            continue
        if re.match(pattern, p, re.DOTALL):
            if cur:
                merged.append(cur)
                cur = ""
            cur = p
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        merged.append(cur)
    # Now recursively split large chunks while keeping small adjacent chunks together
    out = []
    max_chars = 8000
    cur = ""
    for m in merged:
        m = m.strip()
        if not m:
            continue
        if len(cur) + len(m) + 2 <= max_chars and m.startswith("\\"):
            # keep headers with body if possible
            cur = (cur + "\n\n" + m).strip() if cur else m
        else:
            if cur:
                out.extend(split_recursively(cur))
            cur = m
    if cur:
        out.extend(split_recursively(cur))
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: translate_stam_ja_v2.py input.tex output.tex", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    content = open(in_path, "r", encoding="utf-8").read()
    # Keep preamble as is? Preamble contains commands; but translate title/abstract/keywords later via chunks.
    # Split at \begin{document}
    m = re.match(r"(.*?)((\\begin\{document\}).*)", content, re.DOTALL)
    if not m:
        raise SystemExit("Could not find \\begin{document}")
    preamble, body = m.group(1), m.group(2)
    chunks = top_level_chunks(body)
    print(f"Translating {len(chunks)} body chunks...", flush=True)
    translated = [translate_chunk(c) for c in chunks]
    output = preamble + "\n" + "\n".join(translated)
    open(out_path, "w", encoding="utf-8").write(output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
