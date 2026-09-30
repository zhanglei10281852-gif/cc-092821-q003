from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

# 需要纳入规范化的日期格式（统一为 ISO，消除 “2026/9/25” 之类的表示差异）
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y年%m月%d日",
)
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        for fmt in _DATETIME_FORMATS:
            try:
                return datetime.strptime(text, fmt).isoformat()
            except ValueError:
                continue
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
        return text
    return value


def canonicalize(value: Any) -> Any:
    """把请求规范化为稳定结构：键排序、日期/数字表示统一。"""
    value = _normalize_scalar(value)
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def display_value(value: Any) -> Any:
    value = canonicalize(value)
    if isinstance(value, (dict, list)):
        return json.loads(json.dumps(value, ensure_ascii=False))
    return value


def diff_fields(stored: Any, incoming: Any, prefix: str = "") -> list[dict[str, Any]]:
    """列出规范化后仍然不一致的字段（供冲突响应定位差异）。"""
    differences: list[dict[str, Any]] = []
    if isinstance(stored, dict) or isinstance(incoming, dict):
        keys = sorted(set(stored or {}) | set(incoming or {}))
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            if not isinstance(stored, dict) or key not in stored:
                differences.append({"field": name, "original": None, "incoming": display_value((incoming or {}).get(key))})
            elif not isinstance(incoming, dict) or key not in incoming:
                differences.append({"field": name, "original": display_value(stored.get(key)), "incoming": None})
            else:
                differences.extend(diff_fields(stored[key], incoming[key], name))
    elif isinstance(stored, list) or isinstance(incoming, list):
        left, right = stored or [], incoming or []
        if canonical_json(left) != canonical_json(right):
            differences.append({"field": prefix, "original": display_value(stored), "incoming": display_value(incoming)})
    elif canonical_json(stored) != canonical_json(incoming):
        differences.append({"field": prefix, "original": display_value(stored), "incoming": display_value(incoming)})
    return differences
