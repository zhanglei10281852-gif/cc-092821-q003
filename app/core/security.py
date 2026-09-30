from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from app.core.errors import ValidationError

PBKDF2_ITERATIONS = 210_000
MIN_PASSWORD_LENGTH = 10


def normalize_username(value: str) -> str:
    normalized = value.strip().casefold()
    if not 3 <= len(normalized) <= 64:
        raise ValidationError("用户名长度必须在 3 到 64 个字符之间")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-@")
    if any(character not in allowed for character in normalized):
        raise ValidationError("用户名只能包含字母、数字及 . _ - @")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError("密码长度不能少于 10 个字符")
    categories = [
        any(character.islower() for character in password),
        any(character.isupper() for character in password),
        any(character.isdigit() for character in password),
        any(not character.isalnum() for character in password),
    ]
    if sum(categories) < 3:
        raise ValidationError("密码必须包含至少三类字符")


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    validate_password(password)
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), actual_salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(actual_salt).decode().rstrip("="),
        base64.urlsafe_b64encode(digest).decode().rstrip("="),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text + "==")
        expected = base64.urlsafe_b64decode(digest_text + "==")
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def canonicalize(value: Any) -> Any:
    """把请求内容归一成可稳定序列化的形态：键排序、日期统一为 ISO 文本、枚举取其值。

    归一后：JSON 字段顺序不影响结果；date/datetime 与其等价 ISO 字符串表示视为相同内容。
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # 整数值浮点与整数表示（5.0 与 5）视为相同内容。
        return int(value) if value.is_integer() else value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _normalize_datetime_text(value)
    if isinstance(value, dict):
        return {str(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    return value


def _normalize_datetime_text(text: str) -> str:
    """把 ISO 日期/时间字符串归一成与 date/datetime 对象相同的表示；非日期文本原样返回。"""
    if len(text) == 10:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).isoformat()
    return parsed.isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def request_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: int
    username: str
    display_name: str
    department_id: int | None
    permissions: frozenset[str]
    session_id: int

    def can(self, permission: str) -> bool:
        return "*" in self.permissions or permission in self.permissions

    def require(self, permission: str) -> None:
        from app.core.errors import PermissionDeniedError

        if not self.can(permission):
            raise PermissionDeniedError(f"缺少权限：{permission}")
