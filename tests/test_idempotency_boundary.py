from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from app.core.canonical import canonicalize, request_fingerprint
from app.core.clock import to_storage, utc_now
from app.core.errors import IdempotencyConflictError
from app.database import transaction
from app.germplasm.service import GermplasmService


def _prepare_lot_with_pending_schedule(service: GermplasmService) -> tuple[dict, dict]:
    from tests.test_germplasm_workflow import create_stored_lot

    _, lot, _ = create_stored_lot(service)
    protocol = service.viability.create_protocol({
        "protocol_code": "RICE-GER", "crop_name": "水稻", "sample_size": 100, "replicate_count": 2,
        "temperature_c": 25, "duration_days": 14, "normal_seedling_rule": "根芽发育完整",
        "created_by": "技术负责人",
    })
    policy = service.viability.create_policy({
        "crop_name": "水稻", "risk_level": "medium", "interval_months": 12, "warning_days": 30,
        "minimum_germination_percent": 75, "effective_from": "2026-01-01", "effective_to": None,
        "created_by": "技术负责人",
    })
    timestamp = to_storage(utc_now())
    service.connection.execute(
        "INSERT INTO retest_schedules(lot_id,policy_id,due_on,status,reason,created_at,updated_at) "
        "VALUES(?,?,?, 'pending','到期复检',?,?)",
        (lot["id"], policy["id"], "2026-09-20", timestamp, timestamp),
    )
    return lot, protocol


def _payload(lot: dict, protocol: dict, **overrides) -> dict:
    data = {
        "test_no": "VT-100", "lot_id": lot["id"], "protocol_id": protocol["id"], "test_type": "周期复检",
        "sampled_grams": 5, "scheduled_for": "2026-09-25", "requested_by": "检测员",
        "idempotency_key": "schedule-vt-100",
    }
    data.update(overrides)
    return data


def _counts(connection: sqlite3.Connection, lot_id: int, test_no: str) -> tuple[int, int, int]:
    tests = connection.execute("SELECT COUNT(*) FROM viability_tests WHERE lot_id=?", (lot_id,)).fetchone()[0]
    transitions = connection.execute(
        "SELECT COUNT(*) FROM retest_schedules WHERE lot_id=? AND status='scheduled'", (lot_id,)
    ).fetchone()[0]
    intents = connection.execute(
        "SELECT COUNT(*) FROM outbox_events WHERE event_type='viability.sample_intended'"
    ).fetchone()[0]
    return int(tests), int(transitions), int(intents)


def test_identical_retry_replays_first_result(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)

        first = service.viability.schedule_test(payload)
        assert first["idempotency"]["replayed"] is False
        test_id = first["id"]

        retry = service.viability.schedule_test({**payload, "requested_by": "检测员"})
        assert retry["id"] == test_id
        assert retry["idempotency"]["replayed"] is True
        assert retry["sampled_grams"] == 5
        assert retry["scheduled_for"] == "2026-09-25"
        assert _counts(connection, lot["id"], payload["test_no"]) == (1, 1, 1)


def test_changed_detection_fields_conflict_with_original_and_field_list(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)
        first = service.viability.schedule_test(payload)

        for field, changed in {
            "sampled_grams": 9,
            "scheduled_for": "2026-10-01",
            "test_type": "异常复核",
        }.items():
            with pytest.raises(IdempotencyConflictError) as excinfo:
                service.viability.schedule_test(_payload(lot, protocol, **{field: changed}))
            context = excinfo.value.context
            assert context["original"]["test_id"] == first["id"]
            assert context["original"]["test_no"] == "VT-100"
            fields = {item["field"] for item in context["conflicting_fields"]}
            assert field in fields
        # 另一个批次同样冲突
        with pytest.raises(IdempotencyConflictError) as excinfo:
            service.viability.schedule_test(_payload(lot, protocol, lot_id=lot["id"] + 999))
        assert "lot_id" in {item["field"] for item in excinfo.value.context["conflicting_fields"]}

        # 所有冲突请求都没有产生副作用
        assert _counts(connection, lot["id"], payload["test_no"]) == (1, 1, 1)


def test_canonicalization_ignores_key_order_and_date_representation(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)
        first = service.viability.schedule_test(payload)

        reordered = {key: payload[key] for key in reversed(list(payload))}
        for equivalent in (
            reordered,
            _payload(lot, protocol, sampled_grams=5.0, scheduled_for="2026/09/25"),
            _payload(lot, protocol, scheduled_for="2026.09.25"),
        ):
            replay = service.viability.schedule_test(equivalent)
            assert replay["id"] == first["id"]
            assert replay["idempotency"]["replayed"] is True
        assert _counts(connection, lot["id"], payload["test_no"]) == (1, 1, 1)

        # 规范化工具本身：顺序无关、日期表示无关
        assert request_fingerprint({"a": 1, "b": 2}) == request_fingerprint({"b": 2, "a": 1})
        assert canonicalize("2026/9/25") == "2026-09-25"
        assert canonicalize(5.0) == 5


def test_concurrent_creation_with_same_key_produces_single_artifact_set(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)
        lot_id, protocol_id = lot["id"], protocol["id"]

    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            with transaction(immediate=True) as conn:
                results.append(GermplasmService(conn).viability.schedule_test(_payload(
                    {"id": lot_id}, {"id": protocol_id}
                )))
        except Exception as exc:  # noqa: BLE001 - 测试需要收集线程内异常
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(results) == 2
    ids = {item["id"] for item in results}
    assert len(ids) == 1
    assert {item["idempotency"]["replayed"] for item in results} == {True, False}

    with transaction(immediate=True) as connection:
        assert _counts(connection, lot_id, payload["test_no"]) == (1, 1, 1)


def test_semantics_persist_after_restart(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)
        first = service.viability.schedule_test(payload)

    # 模拟服务重启：全新连接重新读库
    from app.database import close_connection
    close_connection()

    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        replay = service.viability.schedule_test(_payload(lot, protocol))
        assert replay["id"] == first["id"]
        assert replay["idempotency"]["replayed"] is True
        with pytest.raises(IdempotencyConflictError):
            service.viability.schedule_test(_payload(lot, protocol, sampled_grams=12))


def test_legacy_record_is_backfilled_and_keeps_semantics(client):
    from app.database import _migrate_idempotency_records

    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        payload = _payload(lot, protocol)
        first = service.viability.schedule_test(payload)
        test_id = first["id"]

        # 改回旧版格式：指纹位置误存 test_no、没有 request_json
        connection.execute(
            "UPDATE idempotency_records SET request_hash=?,request_json=NULL "
            "WHERE scope='viability.schedule' AND idempotency_key=?",
            (payload["test_no"], payload["idempotency_key"]),
        )
        _migrate_idempotency_records(connection)

        row = connection.execute(
            "SELECT request_hash,request_json FROM idempotency_records "
            "WHERE scope='viability.schedule' AND idempotency_key=?",
            (payload["idempotency_key"],),
        ).fetchone()
        stored_request = json.loads(row["request_json"])
        assert stored_request["sampled_grams"] == 5
        assert stored_request["scheduled_for"] == "2026-09-25"

        # 回填后：一致重放、不一致冲突
        assert service.viability.schedule_test(_payload(lot, protocol))["id"] == test_id
        with pytest.raises(IdempotencyConflictError):
            service.viability.schedule_test(_payload(lot, protocol, scheduled_for="2026-11-11"))


def test_api_reports_replay_and_conflict_and_detail_shows_record(client, admin):
    headers = admin["headers"]
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        lot, protocol = _prepare_lot_with_pending_schedule(service)
        lot_id, protocol_id = lot["id"], protocol["id"]

    body = {
        "test_no": "VT-200", "lot_id": lot_id, "protocol_id": protocol_id, "test_type": "周期复检",
        "sampled_grams": 5, "scheduled_for": "2026-09-25", "requested_by": "检测员",
        "idempotency_key": "schedule-vt-200",
    }
    first = client.post("/api/germplasm/tests", json=body, headers=headers)
    assert first.status_code == 201, first.text
    assert first.json()["idempotency"]["replayed"] is False
    test_id = first.json()["id"]

    retry = client.post("/api/germplasm/tests", json=dict(reversed(list(body.items()))), headers=headers)
    assert retry.status_code == 200
    assert retry.json()["id"] == test_id
    assert retry.json()["idempotency"]["replayed"] is True

    conflict = client.post("/api/germplasm/tests", json={**body, "sampled_grams": 8}, headers=headers)
    assert conflict.status_code == 409
    error = conflict.json()["error"]
    assert error["code"] == "idempotency_conflict"
    assert error["context"]["original"]["test_id"] == test_id
    assert any(item["field"] == "sampled_grams" for item in error["context"]["conflicting_fields"])

    detail = client.get(f"/api/germplasm/tests/{test_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["idempotency"]["idempotency_key"] == "schedule-vt-200"
    assert detail.json()["idempotency"]["request_hash"]
