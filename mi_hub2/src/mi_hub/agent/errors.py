"""エラー分類と回復（指示書 §9）。

自動修正は事前定義ルールの範囲内に限定し、意味変更を伴う修正は
人間確認を要求する。自動再試行回数には上限を設ける。
"""

from __future__ import annotations

from typing import Any

from .models import ErrorRecord
from .states import AUTO_RECOVERABLE_ERRORS, HUMAN_REVIEW_ERRORS, ErrorType

# エラーメッセージからの分類キーワード
_KEYWORDS: list[tuple[str, ErrorType]] = [
    ("unit", ErrorType.UNIT_MISMATCH),
    ("単位", ErrorType.UNIT_MISMATCH),
    ("schema", ErrorType.SCHEMA_MISMATCH),
    ("missing input", ErrorType.MISSING_INPUT),
    ("timeout", ErrorType.TIMEOUT),
    ("permission", ErrorType.PERMISSION_ERROR),
    ("network", ErrorType.NETWORK_ERROR),
    ("connection", ErrorType.NETWORK_ERROR),
    ("out of domain", ErrorType.OUT_OF_DOMAIN),
    ("適用範囲外", ErrorType.OUT_OF_DOMAIN),
    ("memory", ErrorType.OUT_OF_MEMORY),
    ("validation", ErrorType.VALIDATION_ERROR),
]


def classify_error(message: str) -> ErrorType:
    low = message.lower()
    for kw, etype in _KEYWORDS:
        if kw in low:
            return etype
    return ErrorType.UNKNOWN_ERROR


def is_auto_recoverable(error_type: ErrorType) -> bool:
    return error_type in AUTO_RECOVERABLE_ERRORS


def requires_human_review(error_type: ErrorType) -> bool:
    return error_type in HUMAN_REVIEW_ERRORS


# 事前定義の単位変換ルール（§9.2 自動修正可能な範囲）
_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("ev/atom", "kj/mol"): 96.485,
    ("kj/mol", "ev/atom"): 1.0 / 96.485,
    ("c", "k"): 1.0,  # オフセット変換は convert_unit 内で処理
}


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """既知の単位ペアのみ変換する。未知ペアは例外を送出し人間確認へ。"""
    f, t = from_unit.strip().lower(), to_unit.strip().lower()
    if f == t:
        return value
    if (f, t) == ("c", "k"):
        return value + 273.15
    if (f, t) == ("k", "c"):
        return value - 273.15
    if (f, t) in _UNIT_CONVERSIONS:
        return value * _UNIT_CONVERSIONS[(f, t)]
    raise ValueError(f"不確実な単位解釈のため人間確認が必要: {from_unit} -> {to_unit}")


def try_auto_fix(
    error_type: ErrorType, inputs: dict[str, Any]
) -> dict[str, Any] | None:
    """自動修正候補を返す。修正できない場合は None（人間確認へ）。"""
    if error_type == ErrorType.UNIT_MISMATCH:
        unit = str(inputs.get("temperature_unit", "")).strip().lower()
        if "temperature" in inputs and unit and unit != "k":
            try:
                fixed = dict(inputs)
                fixed["temperature"] = convert_unit(float(inputs["temperature"]), unit, "K")
                fixed["temperature_unit"] = "K"
                return fixed
            except (ValueError, TypeError):
                return None
        return None
    if error_type == ErrorType.SCHEMA_MISMATCH:
        # 入力キーの別名変換（事前定義ルール）
        aliases = {"temp": "temperature", "comp": "composition", "T": "temperature"}
        fixed = {aliases.get(k, k): v for k, v in inputs.items()}
        return fixed if fixed != inputs else None
    if error_type in (ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT):
        return dict(inputs)  # 入力そのままで再試行
    return None


def record_error(task_id: str | None, message: str) -> ErrorRecord:
    etype = classify_error(message)
    return ErrorRecord(
        task_id=task_id,
        error_type=etype,
        message=message,
        auto_recoverable=is_auto_recoverable(etype),
    )
