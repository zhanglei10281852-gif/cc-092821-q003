from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from app.core.canonical import canonicalize, diff_fields, request_fingerprint
from app.core.clock import Clock, SystemClock, to_storage
from app.core.errors import IdempotencyConflictError


@dataclass(frozen=True, slots=True)
class StoredResponse:
    body: dict
    status_code: int
    replayed: bool


class IdempotencyService:
    """以 (scope,key) 为业务键，只允许内容完全一致（规范化后）的请求重放。"""

    def __init__(self, connection: sqlite3.Connection, clock: Clock | None = None) -> None:
        self.connection = connection
        self.clock = clock or SystemClock()

    def stored(self, scope: str, key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM idempotency_records WHERE scope=? AND idempotency_key=?", (scope, key)
        ).fetchone()

    def check(self, scope: str, key: str, payload: Any) -> StoredResponse | None:
        """命中且内容一致 → 返回首次结果；内容不一致 → 抛出 IdempotencyConflictError。"""
        row = self.stored(scope, key)
        if row is None:
            return None
        canonical = canonicalize(payload)
        if request_fingerprint(canonical) != row["request_hash"]:
            raise IdempotencyConflictError(
                "同一幂等键对应了不同的请求内容，不能重放为首次结果",
                context={
                    "idempotency_key": key,
                    "original": json.loads(row["response_json"]),
                    "conflicting_fields": diff_fields(self._stored_payload(row), canonical),
                },
            )
        return StoredResponse(json.loads(row["response_json"]), int(row["status_code"]), True)

    @staticmethod
    def _stored_payload(row: sqlite3.Row) -> Any:
        if "request_json" in row.keys() and row["request_json"]:
            try:
                return canonicalize(json.loads(row["request_json"]))
            except json.JSONDecodeError:
                pass
        # 服务升级前写入的旧记录没有留存请求体
        return {"request_fingerprint": "<旧记录未留存请求内容>"}

    def save(self, scope: str, key: str, payload: Any, body: dict, status_code: int) -> StoredResponse:
        canonical = canonicalize(payload)
        try:
            self.connection.execute(
                "INSERT INTO idempotency_records(scope,idempotency_key,request_hash,request_json,response_json,"
                "status_code,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    scope, key, request_fingerprint(canonical),
                    json.dumps(canonical, ensure_ascii=False, sort_keys=True),
                    json.dumps(body, ensure_ascii=False, sort_keys=True), status_code, to_storage(self.clock.now()),
                ),
            )
        except sqlite3.IntegrityError:
            existing = self.check(scope, key, payload)
            if existing is None:
                raise
            return existing
        return StoredResponse(body, status_code, False)

    def claim(self, scope: str, key: str, payload: Any) -> StoredResponse | None:
        """在执行业务前先占用业务键。

        返回 None 表示占位成功（事务回退时占位记录一并消失）；
        若键已被并发事务提交，则退化为 check：内容一致返回首次结果，不一致抛冲突。
        """
        canonical = canonicalize(payload)
        timestamp = to_storage(self.clock.now())
        try:
            self.connection.execute(
                "INSERT INTO idempotency_records(scope,idempotency_key,request_hash,request_json,response_json,"
                "status_code,created_at) VALUES(?,?,?,?,?,0,?)",
                (
                    scope, key, request_fingerprint(canonical),
                    json.dumps(canonical, ensure_ascii=False, sort_keys=True), "{}", timestamp,
                ),
            )
        except sqlite3.IntegrityError:
            return self.check(scope, key, payload)
        return None

    def finalize(self, scope: str, key: str, body: dict, status_code: int) -> None:
        """业务执行成功后写入首次响应，与业务数据在同一事务内提交。"""
        self.connection.execute(
            "UPDATE idempotency_records SET response_json=?,status_code=?,created_at=? WHERE scope=? AND idempotency_key=?",
            (json.dumps(body, ensure_ascii=False, sort_keys=True), status_code, to_storage(self.clock.now()), scope, key),
        )

    def execute(
        self,
        scope: str,
        key: str,
        payload: Any,
        operation: Callable[[], tuple[dict, int]],
    ) -> StoredResponse:
        existing = self.check(scope, key, payload)
        if existing is not None:
            return existing
        body, status_code = operation()
        return self.save(scope, key, payload, body, status_code)
