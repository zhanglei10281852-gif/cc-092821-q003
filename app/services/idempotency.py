from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from app.core.clock import Clock, SystemClock, to_storage
from app.core.errors import ConflictError
from app.core.security import canonical_json, canonicalize, request_fingerprint

# 初始占位记录：并发请求只有一个能 INSERT 成功，保证副作用只发生一次。
PENDING_HASH = "__pending__"


class IdempotencyConflictError(ConflictError):
    """同一幂等键被用于内容不同的请求。"""

    def __init__(
        self,
        message: str,
        *,
        conflicts: list[dict[str, str]] | None = None,
        original_resource: dict[str, Any] | None = None,
        status: str = "replayed",
    ) -> None:
        super().__init__(message, context={
            "status": status,
            "original_resource": original_resource or {},
            "conflicts": conflicts or [],
        })
        self.conflicts = conflicts or []
        self.original_resource = original_resource or {}
        self.status = status


@dataclass(frozen=True, slots=True)
class StoredResponse:
    body: dict
    status_code: int
    replayed: bool


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    scope: str
    key: str
    request_hash: str
    request_snapshot: dict[str, Any]
    resource: dict[str, Any]
    response: dict[str, Any]
    status_code: int
    created_at: str
    status: str  # replayed / pending

    @property
    def completed(self) -> bool:
        return self.request_hash != PENDING_HASH


class IdempotencyService:
    def __init__(self, connection: sqlite3.Connection, clock: Clock | None = None) -> None:
        self.connection = connection
        self.clock = clock or SystemClock()

    # ------------------------------------------------------------------ 查询

    def get_record(self, scope: str, key: str) -> IdempotencyRecord | None:
        row = self.connection.execute(
            "SELECT * FROM idempotency_records WHERE scope=? AND idempotency_key=?", (scope, key)
        ).fetchone()
        return None if row is None else self._to_record(row)

    def lookup(self, scope: str, key: str, payload: dict) -> StoredResponse | None:
        """命中且内容一致才回放；内容不同抛出带冲突字段的 409。"""
        record = self.get_record(scope, key)
        if record is None or not record.completed:
            return None
        self._assert_matches(record, payload)
        return StoredResponse(record.response, record.status_code, True)

    def heal(
        self, scope: str, key: str, payload: dict, resource: dict[str, Any] | None = None
    ) -> None:
        """把升级前没有请求快照的旧记录补齐为严格指纹，服务重启后语义不变。"""
        self.connection.execute(
            "UPDATE idempotency_records SET request_hash=?,request_json=?,resource_json=? "
            "WHERE scope=? AND idempotency_key=?",
            (
                request_fingerprint(payload), canonical_json(payload),
                json.dumps(resource or {}, ensure_ascii=False, sort_keys=True), scope, key,
            ),
        )

    def find_by_test_id(self, test_id: int, scope: str | None = None) -> IdempotencyRecord | None:
        sql = (
            "SELECT * FROM idempotency_records WHERE ("
            "json_extract(resource_json,'$.test_id')=? "
            "OR (resource_json='{}' AND json_extract(response_json,'$.test_id')=?))"
        )
        params: list[Any] = [test_id, test_id]
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        sql += " ORDER BY created_at DESC,idempotency_key LIMIT 1"
        row = self.connection.execute(sql, params).fetchone()
        return None if row is None else self._to_record(row)

    def describe(self, record: IdempotencyRecord) -> dict[str, Any]:
        return {
            "scope": record.scope,
            "idempotency_key": record.key,
            "status": record.status,
            "request_hash": record.request_hash if record.completed else None,
            "request": record.request_snapshot,
            "resource": record.resource,
            "status_code": record.status_code if record.completed else None,
            "created_at": record.created_at,
        }

    def _to_record(self, row: sqlite3.Row) -> IdempotencyRecord:
        request_hash = row["request_hash"]
        return IdempotencyRecord(
            scope=row["scope"],
            key=row["idempotency_key"],
            request_hash=request_hash,
            request_snapshot=self._load_json(row["request_json"]),
            resource=self._load_json(row["resource_json"]),
            response=self._load_json(row["response_json"]),
            status_code=int(row["status_code"]),
            created_at=row["created_at"],
            status="pending" if request_hash == PENDING_HASH else "replayed",
        )

    # --------------------------------------------------------------- 保留/完成

    def reserve(self, scope: str, key: str, payload: dict) -> IdempotencyRecord | None:
        """为当前请求占位。返回 None 表示占位成功（调用方执行操作）；
        返回已存在记录表示被其他请求占用，需要进一步比对内容。
        """
        fingerprint = request_fingerprint(payload)
        timestamp = to_storage(self.clock.now())
        try:
            self.connection.execute(
                "INSERT INTO idempotency_records(scope,idempotency_key,request_hash,request_json,response_json,"
                "resource_json,status_code,created_at) VALUES(?,?,?,?,?,?,0,?)",
                (scope, key, PENDING_HASH, canonical_json(payload), "{}", "{}", timestamp),
            )
        except sqlite3.IntegrityError:
            existing = self.get_record(scope, key)
            if existing is None:  # 理论上不会发生：插入失败必然有记录
                raise
            if existing.completed:
                self._assert_matches(existing, payload)
                return existing
            # 另一个并发请求仍在执行：内容指纹一致才允许等待，否则直接拒绝。
            if existing.request_snapshot and request_fingerprint(existing.request_snapshot) != fingerprint:
                raise IdempotencyConflictError(
                    "同一幂等键已有进行中的不同请求",
                    conflicts=diff_fields(existing.request_snapshot, canonicalize(payload)),
                    original_resource=existing.resource,
                    status="pending",
                )
            return existing
        return None

    def complete(
        self,
        scope: str,
        key: str,
        payload: dict,
        body: dict,
        status_code: int,
        resource: dict[str, Any] | None = None,
    ) -> StoredResponse:
        fingerprint = request_fingerprint(payload)
        self.connection.execute(
            "UPDATE idempotency_records SET request_hash=?,request_json=?,response_json=?,resource_json=?,"
            "status_code=? WHERE scope=? AND idempotency_key=?",
            (
                fingerprint, canonical_json(payload),
                json.dumps(body, ensure_ascii=False, sort_keys=True),
                json.dumps(resource or {}, ensure_ascii=False, sort_keys=True),
                status_code, scope, key,
            ),
        )
        return StoredResponse(body, status_code, False)

    def release(self, scope: str, key: str) -> None:
        """操作失败时清掉自己的占位记录，允许客户端修正后用新键或重试。"""
        self.connection.execute(
            "DELETE FROM idempotency_records WHERE scope=? AND idempotency_key=? AND request_hash=?",
            (scope, key, PENDING_HASH),
        )

    # ----------------------------------------------------------------- 执行器

    def execute(self, scope: str, key: str, payload: dict, operation: Callable[[], tuple[dict, int]]) -> StoredResponse:
        existing = self.lookup(scope, key, payload)
        if existing is not None:
            return existing
        holder = self.reserve(scope, key, payload)
        if holder is not None:
            if not holder.completed:
                raise ConflictError("同一幂等键的请求仍在处理中，请稍后重试")
            return StoredResponse(holder.response, holder.status_code, True)
        try:
            body, status_code = operation()
        except Exception:
            self.release(scope, key)
            raise
        return self.complete(scope, key, payload, body, status_code)

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _load_json(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _assert_matches(record: IdempotencyRecord, payload: dict) -> None:
        fingerprint = request_fingerprint(payload)
        if record.request_hash == fingerprint:
            return
        snapshot = record.request_snapshot
        conflicts = diff_fields(snapshot, canonicalize(payload)) if snapshot else []
        raise IdempotencyConflictError(
            "同一幂等键不能用于内容不同的请求",
            conflicts=conflicts,
            original_resource=record.resource,
            status=record.status,
        )


def diff_fields(before: Any, after: Any, path: str = "") -> list[dict[str, str]]:
    """返回规范化后不一致的字段路径及双方取值（只列叶子差异）。"""
    if isinstance(before, dict) and isinstance(after, dict):
        diffs: list[dict[str, str]] = []
        for name in sorted(set(before) | set(after)):
            child = f"{path}.{name}" if path else str(name)
            if name not in before or name not in after:
                diffs.append({"field": child, "original": _preview(before.get(name)), "conflicting": _preview(after.get(name))})
            else:
                diffs.extend(diff_fields(before[name], after[name], child))
        return diffs
    if before != after:
        return [{"field": path or "$", "original": _preview(before), "conflicting": _preview(after)}]
    return []


def _preview(value: Any, limit: int = 200) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        return "null"
    else:
        text = canonical_json(value) if isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"
