from __future__ import annotations

import json
from typing import Any

import pandas as pd


def parse_trigger_reasons(value: Any) -> list[str]:
    """
    Parse stored trigger reasons from JSON/string/list.
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [text]


def format_price(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{int(float(value)):,}"
    except Exception:
        return "-"


def format_pct(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "-"