"""Intent classifier: detect out-of-scope queries before LLM SQL generation.

Classifies queries into:
- db_query: answerable by the materials database (pass to SQL pipeline)
- out_of_scope: VASP workflow, DFT methodology, general knowledge
- ambiguous: needs clarification before proceeding
- unsafe: SQL injection or destructive intent
- greeting: social/chat messages
"""
from __future__ import annotations

import re
from typing import Any


# Patterns that strongly indicate VASP/DFT workflow questions (NOT DB queries)
_VASP_WORKFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"INCAR|KPOINTS|POSCAR|POTCAR|CONTCAR|OUTCAR|CHGCAR|WAVECAR|DOSCAR|PROCAR", re.I),
    re.compile(r"(?:SCF|ionic|electronic)\s*(?:convergence|loop|step|収束)", re.I),
    re.compile(r"(?:ENCUT|EDIFF|EDIFFG|ISMEAR|SIGMA|NBANDS|ISPIN|LORBIT|LREAL|PREC|ALGO|IBRION|ISIF|NSW|NELM)(?:\b|[をはがのにで])", re.I),
    re.compile(r"VASP(?:で|の|を|は|に|settings|input|run|calculation|tutorial|manual)", re.I),
    re.compile(r"(?:HSE|PBE|GGA|mBJ|SOC|DFT)\s*(?:計算|calculation|run|settings|setup|手順|procedure)", re.I),
    re.compile(r"(?:フォノン|phonon)\s*(?:計算|dispersion|band|spectrum|安定|不安定|虚数)", re.I),
    re.compile(r"Wannier(?:90|化|ization|interpolation)", re.I),
    re.compile(r"(?:Bader|charge|analysis)\s*(?:charge|analysis|解析)", re.I),
    re.compile(r"(?:バンド構造|band\s*structure)\s*(?:計算|compute|plot|描く|を)", re.I),
    re.compile(r"(?:状態密度|DOS)\s*(?:計算|compute|plot|を)", re.I),
    re.compile(r"(?:有効質量|effective\s*mass)\s*(?:計算|compute|求め|出し)", re.I),
    re.compile(r"(?:how|手順|教えて|方法|procedure|tutorial)\s*(?:to|で|は)", re.I),
    re.compile(r"(?:なぜ|why|reason|理由)\s*(?:違う|different|変わる|change)", re.I),
    re.compile(r"(?:収束しない|not\s*converge|convergence\s*(?:issue|problem|fail))", re.I),
    re.compile(r"(?:どう|how)\s*(?:読む|read|interpret|解釈)", re.I),
    re.compile(r"(?:VBM|CBM|Fermi\s*energy|フェルミ)\s*(?:どこ|where|読む|read)", re.I),
    re.compile(r"(?:partial\s*occupancy|部分占有)", re.I),
    re.compile(r"(?:imaginary\s*(?:mode|frequency)|虚数振動)", re.I),
    re.compile(r"(?:dielectric|誘電)\s*(?:constant|定数|tensor|テンソル)", re.I),
    re.compile(r"topological", re.I),
]

# DB-specific patterns: these strongly indicate a DB query
_DB_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:を|だけ|全部|全て|リスト|一覧|出して|見たい|探して|検索|ある[？?]?)$", re.I),
    re.compile(r"(?:B2|L1[2₂]|CsCl|Cu3Au)\s*(?:化合物|compound|alloy|合金|構造|エントリ|entry)", re.I),
    re.compile(r"(?:formation\s*energy|形成エネルギー|band\s*gap|バンドギャップ|lattice\s*constant|格子定数)", re.I),
    re.compile(r"(?:energy\s*above\s*hull|Ehull|E_hull)", re.I),
    re.compile(r"(?:安定|stable|metastable|準安定)\s*(?:な|の|化合物|compound)", re.I),
    re.compile(r"(?:[A-Z][a-z]?)\s*(?:を含む|含む|containing|with|系)", re.I),
    re.compile(r"(?:順|sort|order|並べ|ランキング|top)\s*(?:に|で|by)", re.I),
    re.compile(r"entry_id|formula|prototype|strukturbericht", re.I),
    re.compile(r"(?:>[<=]?\s*\d|<[>=]?\s*\d|\d\s*(?:eV|Å|GPa))", re.I),
]

# Greeting/chat patterns
_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:こんにちは|こんばんは|おはよう|hello|hi|hey|good\s*(?:morning|afternoon|evening))[\s!！。.]*$", re.I),
    re.compile(r"^(?:ありがとう|thank|thanks)[\s!！。.]*$", re.I),
    re.compile(r"(?:今日の天気|weather|what\s*time)", re.I),
]

# Unsafe patterns (SQL injection attempts)
_UNSAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE)\s+", re.I),
    re.compile(r";\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER)", re.I),
    re.compile(r"(?:system|admin|root)\s*(?:secret|password|credential)", re.I),
    re.compile(r"(?:show|dump|list|get|extract)\s*(?:secret|password|credential)", re.I),
]

# Properties not in the DB schema
_OUT_OF_SCHEMA_PROPERTIES: list[re.Pattern[str]] = [
    re.compile(r"(?:shear\s*modulus|剪断弾性率)", re.I),
    re.compile(r"(?:Young'?s?\s*modulus|ヤング率)", re.I),
    re.compile(r"(?:Poisson'?s?\s*ratio|ポアソン比)", re.I),
    re.compile(r"(?:thermal\s*conductivity|熱伝導率)", re.I),
    re.compile(r"(?:magnetic\s*moment|磁気モーメント|磁性)", re.I),
    re.compile(r"(?:direct\s*gap|indirect\s*gap)", re.I),
]


def classify_intent(query: str) -> dict[str, Any]:
    """Classify the intent of a natural-language query.

    Returns a dict with:
      - intent: one of db_query, out_of_scope, ambiguous, unsafe, greeting
      - confidence: float 0-1
      - reason: human-readable explanation
      - matched_patterns: list of pattern names that matched
    """
    query = query.strip()
    if not query:
        return {
            "intent": "out_of_scope",
            "confidence": 1.0,
            "reason": "Empty query",
            "matched_patterns": [],
        }

    matched_vasp: list[str] = []
    matched_db: list[str] = []
    matched_greeting: list[str] = []
    matched_unsafe: list[str] = []
    matched_oos_prop: list[str] = []

    for p in _VASP_WORKFLOW_PATTERNS:
        m = p.search(query)
        if m:
            matched_vasp.append(m.group())

    for p in _DB_QUERY_PATTERNS:
        m = p.search(query)
        if m:
            matched_db.append(m.group())

    for p in _GREETING_PATTERNS:
        m = p.search(query)
        if m:
            matched_greeting.append(m.group())

    for p in _UNSAFE_PATTERNS:
        m = p.search(query)
        if m:
            matched_unsafe.append(m.group())

    for p in _OUT_OF_SCHEMA_PROPERTIES:
        m = p.search(query)
        if m:
            matched_oos_prop.append(m.group())

    # Decision logic
    if matched_unsafe:
        return {
            "intent": "unsafe",
            "confidence": 0.95,
            "reason": f"Potentially unsafe input: {', '.join(matched_unsafe)}",
            "matched_patterns": matched_unsafe,
        }

    if matched_greeting and not matched_db:
        return {
            "intent": "greeting",
            "confidence": 0.9,
            "reason": f"Social/chat message: {', '.join(matched_greeting)}",
            "matched_patterns": matched_greeting,
        }

    # VASP workflow vs DB query
    vasp_score = len(matched_vasp)
    db_score = len(matched_db)

    if vasp_score > 0 and db_score == 0:
        return {
            "intent": "out_of_scope",
            "confidence": min(0.5 + vasp_score * 0.15, 0.95),
            "reason": f"VASP/DFT workflow question (not a database query): {', '.join(matched_vasp[:3])}",
            "matched_patterns": matched_vasp,
        }

    if vasp_score > 0 and db_score > 0:
        if vasp_score >= db_score:
            return {
                "intent": "out_of_scope",
                "confidence": 0.6,
                "reason": f"Likely workflow question despite DB terms: VASP={vasp_score} vs DB={db_score}",
                "matched_patterns": matched_vasp + matched_db,
            }
        return {
            "intent": "db_query",
            "confidence": 0.7,
            "reason": f"DB query with some workflow terms: DB={db_score} vs VASP={vasp_score}",
            "matched_patterns": matched_db + matched_vasp,
        }

    if matched_oos_prop and db_score <= 1:
        return {
            "intent": "out_of_scope",
            "confidence": 0.7,
            "reason": f"Property not in database schema: {', '.join(matched_oos_prop)}",
            "matched_patterns": matched_oos_prop,
        }

    if db_score > 0:
        return {
            "intent": "db_query",
            "confidence": min(0.5 + db_score * 0.1, 0.95),
            "reason": f"Database query: {', '.join(matched_db[:3])}",
            "matched_patterns": matched_db,
        }

    # Fallback: short queries with no matches
    if len(query) < 10:
        return {
            "intent": "ambiguous",
            "confidence": 0.5,
            "reason": "Very short query with no clear intent",
            "matched_patterns": [],
        }

    return {
        "intent": "db_query",
        "confidence": 0.4,
        "reason": "No strong signal; defaulting to DB query",
        "matched_patterns": [],
    }


# --- Query Type Classification ---
# Determines the SELECT structure: list individual rows vs. aggregate

_COUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:何件|件数|総数|数え|数を|カウント|count)", re.I),
]

_RATIO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:割合|比率|percentage|ratio)", re.I),
]

# Only trigger GROUP BY when the question explicitly asks for aggregated
# statistics (average, sum) — NOT for "ごとに整理" or "分類" which in
# materials science context typically expect individual rows.
_AGGREGATE_PATTERNS_STRICT: list[re.Pattern[str]] = [
    re.compile(r"(?:平均|average|avg|mean)\s*(?:を|の|は|値)", re.I),
    re.compile(r"(?:合計|total|sum)\s*(?:を|の|は)", re.I),
    re.compile(r"(?:統計|statistics|ヒストグラム|histogram)", re.I),
]

_TOP_N_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:最も多い|最も少ない|most|least|最頻)", re.I),
]


def classify_query_type(query: str) -> dict[str, Any]:
    """Classify the expected output structure of a DB query.

    Conservative strategy: default to individual rows. Only use GROUP BY
    when the question explicitly asks for counts, averages, or ratios.
    Words like "ごとに整理", "分類", "分布" in materials science context
    typically expect individual rows with relevant columns for sorting.

    Returns:
      - query_type: "list" | "aggregate" | "count" | "top_n"
      - instruction: prompt instruction for the LLM
    """
    for pat in _COUNT_PATTERNS:
        if pat.search(query):
            return {
                "query_type": "count",
                "instruction": "Return a COUNT query. Do NOT list individual rows.",
            }

    for pat in _RATIO_PATTERNS:
        if pat.search(query):
            return {
                "query_type": "aggregate",
                "instruction": "Return a ratio/percentage using COUNT with FILTER or CASE. Do NOT list individual rows.",
            }

    for pat in _TOP_N_PATTERNS:
        if pat.search(query):
            return {
                "query_type": "top_n",
                "instruction": "Use GROUP BY + ORDER BY + LIMIT to find the top/bottom items.",
            }

    for pat in _AGGREGATE_PATTERNS_STRICT:
        if pat.search(query):
            return {
                "query_type": "aggregate",
                "instruction": "Use aggregate functions (AVG, COUNT, SUM) with GROUP BY as appropriate.",
            }

    # Default: list individual rows (most common and safest).
    # "ごとに整理", "分類", "分布", "傾向" all return individual rows
    # with ORDER BY for organization. Do NOT use GROUP BY.
    return {
        "query_type": "list",
        "instruction": (
            "Return individual rows. Do NOT use GROUP BY or aggregate functions. "
            "If the question mentions organizing by category (ごとに, 分類, 分布), "
            "include the category column and use ORDER BY to organize results. "
            "Return only columns directly relevant to the question."
        ),
    }
