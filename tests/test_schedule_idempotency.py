from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date

import pytest

from app.core.errors import ConflictError
from app.core.security import request_fingerprint
from app.database import transaction
from app.germplasm.service import GermplasmService
from app.germplasm.viability import SCHEDULE_SCOPE
from app.services.idempotency import IdempotencyService
from tests.test_germplasm_workflow import create_stored_lot


def _protocol(service: GermplasmService) -> dict:
    return service.viability.create_protocol({
        "protocol_code": "RICE-GER", "crop_name": "水稻", "sample_size": 100, "replicate_count": 2,
        "temperature_c": 25, "duration_days": 14, "normal_seedling_rule": "根芽发育完整",
        "created_by": "技术负责人",
    })


def _another_protocol(service: GermplasmService) -> dict:
    return service.viability.create_protocol({
        "protocol_code": "RICE-GER2", "crop_name": "水稻", "sample_size": 100, "replicate_count": 2,
        "temperature_c": 25, "duration_days": 14, "normal_seedling_rule": "根芽发育完整",
        "created_by": "技术负责人",
    })


def _policy(service: GermplasmService) -> dict:
    return service.viability.create_policy({
        "crop_name": "水稻", "risk_level": "medium", "interval_months": 12, "warning_days": 30,
        "minimum_germination_percent": 75, "effective_from": "2026-01-01", "effective_to": None,
        "created_by": "技术负责人",
    })


def _schedule_payload(lot: dict, protocol: dict, **overrides) -> dict:
    payload = {
        "test_no": "VT-IDEM-001", "lot_id": lot["id"], "protocol_id": protocol["id"],
        "test_type": "周期复检", "sampled_grams": 5, "scheduled_for": "2026-09-25",
        "requested_by": "检测员", "idempotency_key": "schedule-idem-0001",
    }
    payload.update(overrides)
    return payload


def _counts(connection: sqlite3.Connection) -> tuple[int, int, int]:
    tests = connection.execute("SELECT COUNT(*) FROM viability_tests").fetchone()[0]
    intents = connection.execute(
        "SELECT COUNT(*) FROM outbox_events WHERE event_type='viability.sample_intended'"
    ).fetchone()[0]
    scheduled = connection.execute(
        "SELECT COUNT(*) FROM retest_schedules WHERE status='scheduled'"
    ).fetchone()[0]
    return int(tests), int(intents), int(scheduled)


def _jsonable(payload: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in payload.items()}


def test_safe_retry_with_same_content_returns_first_result(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "101")
        protocol = _protocol(service)
        first = service.viability.schedule_test(_schedule_payload(lot, protocol))
        assert first["idempotency"]["status"] == "created"

        # 调整字段顺序、日期用 date 对象、取样量用整数浮点——内容一致必须回放。
        retry = {
            "idempotency_key": "schedule-idem-0001", "requested_by": "检测员",
            "scheduled_for": date(2026, 9, 25), "sampled_grams": 5.0,
            "test_type": "周期复检", "protocol_id": protocol["id"], "lot_id": lot["id"],
            "test_no": "VT-IDEM-001",
        }
        second = service.viability.schedule_test(retry)
        assert second["id"] == first["id"]
        assert second["idempotency"]["status"] == "replayed"
        assert _counts(connection) == (1, 1, 0)


def test_changed_detection_field_is_rejected_with_original_and_conflicts(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "102")
        protocol = _protocol(service)
        first = service.viability.schedule_test(_schedule_payload(lot, protocol))
        other_protocol = _another_protocol(service)

        cases = {
            "lot_id": 999999,
            "sampled_grams": 9,
            "scheduled_for": "2026-10-01",
            "test_type": "异常复核",
            "protocol_id": other_protocol["id"],
            "test_no": "VT-IDEM-002",
            "requested_by": "另一位检测员",
        }
        for field, new_value in cases.items():
            with pytest.raises(ConflictError) as exc_info:
                service.viability.schedule_test(_schedule_payload(lot, protocol, **{field: new_value}))
            context = exc_info.value.context
            assert context["original_resource"]["test_id"] == first["id"]
            assert context["original_resource"]["test_no"] == "VT-IDEM-001"
            assert context["status"] == "replayed"
            assert [item["field"] for item in context["conflicts"]] == [field]
            assert context["conflicts"][0]["original"] != context["conflicts"][0]["conflicting"]

        # 被拒绝后取样单没有任何变化：仍是首次任务，副作用各只有一次。
        assert _counts(connection) == (1, 1, 0)


def test_rejection_does_not_consume_key_and_safe_replay_still_works(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "103")
        protocol = _protocol(service)
        first = service.viability.schedule_test(_schedule_payload(lot, protocol))

        with pytest.raises(ConflictError):
            service.viability.schedule_test(_schedule_payload(lot, protocol, sampled_grams=8))

        again = service.viability.schedule_test(_schedule_payload(lot, protocol))
        assert again["id"] == first["id"]
        assert again["idempotency"]["status"] == "replayed"
        assert _counts(connection) == (1, 1, 0)


def test_failed_first_request_releases_reservation(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "104")
        protocol = _protocol(service)
        # 取样量超过批次可用重量：首次请求失败，不应留下占位记录。
        with pytest.raises(ConflictError):
            service.viability.schedule_test(_schedule_payload(lot, protocol, sampled_grams=99999))
        record = IdempotencyService(connection).get_record(SCHEDULE_SCOPE, "schedule-idem-0001")
        assert record is None
        # 同键修正为合法内容后应能创建成功。
        created = service.viability.schedule_test(_schedule_payload(lot, protocol))
        assert created["idempotency"]["status"] == "created"
        assert _counts(connection) == (1, 1, 0)


def test_retest_schedule_transitions_exactly_once(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "105")
        protocol = _protocol(service)
        timestamp = "2026-09-01T00:00:00+00:00"
        connection.execute(
            "INSERT INTO retest_schedules(lot_id,policy_id,due_on,status,reason,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (lot["id"], _policy(service)["id"], "2026-09-20", "pending", "到期复检", timestamp, timestamp),
        )
        first = service.viability.schedule_test(_schedule_payload(lot, protocol))
        service.viability.schedule_test(_schedule_payload(lot, protocol))  # 安全重试
        rows = connection.execute(
            "SELECT status FROM retest_schedules WHERE lot_id=?", (lot["id"],)
        ).fetchall()
        assert [row[0] for row in rows] == ["scheduled"]
        intent = json.loads(connection.execute(
            "SELECT payload_json FROM outbox_events WHERE event_type='viability.sample_intended'"
        ).fetchone()[0])
        assert intent["test_id"] == first["id"]
        assert intent["retest_schedule_id"] is not None
        assert _counts(connection)[:2] == (1, 1)


def test_reservation_ensures_concurrent_identical_create_is_single_task(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "107")
        protocol = _protocol(service)
        idempotency = IdempotencyService(connection)
        payload = _schedule_payload(lot, protocol)
        canonical = {k: v for k, v in payload.items() if k != "idempotency_key"}

        assert idempotency.reserve(SCHEDULE_SCOPE, payload["idempotency_key"], canonical) is None
        holder = idempotency.reserve(SCHEDULE_SCOPE, payload["idempotency_key"], canonical)
        assert holder is not None and not holder.completed  # 并发的相同请求只能等待

        created = service.viability._create_test(payload)
        idempotency.complete(
            SCHEDULE_SCOPE, payload["idempotency_key"], canonical,
            {"test_id": created["id"], "test_no": created["test_no"]}, 201,
            resource={"resource_type": "viability_test", "test_id": created["id"], "test_no": created["test_no"]},
        )
        replayed = idempotency.lookup(SCHEDULE_SCOPE, payload["idempotency_key"], canonical)
        assert replayed is not None
        assert replayed.body["test_id"] == created["id"]
        assert _counts(connection) == (1, 1, 0)


def test_concurrent_different_content_is_rejected_while_pending(client):
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "111")
        protocol = _protocol(service)
        idempotency = IdempotencyService(connection)
        payload = _schedule_payload(lot, protocol)
        canonical = {k: v for k, v in payload.items() if k != "idempotency_key"}
        idempotency.reserve(SCHEDULE_SCOPE, payload["idempotency_key"], canonical)

        with pytest.raises(ConflictError) as exc_info:
            idempotency.reserve(
                SCHEDULE_SCOPE, payload["idempotency_key"], {**canonical, "sampled_grams": 12}
            )
        assert exc_info.value.context["status"] == "pending"
        assert [c["field"] for c in exc_info.value.context["conflicts"]] == ["sampled_grams"]


def test_parallel_identical_requests_across_connections_create_one_task(client):
    """两个独立连接并发提交相同请求：BEGIN IMMEDIATE 串行化，只产生一个检测任务。"""
    from app.database import _create_connection

    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "112")
        protocol = _protocol(service)
        lot_id, protocol_id = lot["id"], protocol["id"]

    outcome: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        conn = _create_connection()
        try:
            barrier.wait(timeout=5)
            conn.execute("BEGIN IMMEDIATE")
            try:
                svc = GermplasmService(conn)
                detail = svc.viability.schedule_test({
                    "test_no": "VT-CONC-001", "lot_id": lot_id, "protocol_id": protocol_id,
                    "test_type": "周期复检", "sampled_grams": 5, "scheduled_for": "2026-09-25",
                    "requested_by": "检测员", "idempotency_key": "schedule-conc-001",
                })
                conn.commit()
                outcome.append({"test_id": detail["id"], "status": detail["idempotency"]["status"]})
            except Exception:
                conn.rollback()
                raise
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(outcome) == 2
    assert len({item["test_id"] for item in outcome}) == 1
    assert {item["status"] for item in outcome} == {"created", "replayed"}
    with transaction(immediate=True) as connection:
        assert _counts(connection) == (1, 1, 0)


def test_idempotency_info_and_api_visibility(client, admin):
    headers = admin["headers"]
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "108")
        protocol = _protocol(service)
        created = service.viability.schedule_test(_schedule_payload(lot, protocol))
        test_id = created["id"]

    detail = client.get(f"/api/germplasm/tests/{test_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    record = detail.json()["idempotency"]
    assert record["idempotency_key"] == "schedule-idem-0001"
    assert record["status"] == "replayed"
    assert record["request"]["sampled_grams"] == 5
    assert record["request"]["scheduled_for"] == "2026-09-25"
    assert record["resource"]["test_id"] == test_id

    record_response = client.get(f"/api/germplasm/tests/{test_id}/idempotency", headers=headers)
    assert record_response.status_code == 200, record_response.text
    assert record_response.json()["record"]["status"] == "replayed"


def test_api_conflict_response_shape(client, admin):
    headers = admin["headers"]
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "109")
        protocol = _protocol(service)
        created = service.viability.schedule_test(_schedule_payload(lot, protocol))
        test_id = created["id"]

    changed = _schedule_payload(lot, protocol, scheduled_for="2026-10-01")
    response = client.post("/api/germplasm/tests", headers=headers, json=_jsonable(changed))
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "conflict"
    assert error["context"]["original_resource"]["test_id"] == test_id
    assert error["context"]["conflicts"][0]["field"] == "scheduled_for"


def test_request_fingerprint_is_order_and_repr_independent():
    from app.core.security import canonicalize

    a = {"b": 1, "a": {"y": [1, 2], "x": "2026-09-25"}}
    b = {"a": {"x": date(2026, 9, 25), "y": [1.0, 2]}, "b": 1}
    assert request_fingerprint(a) == request_fingerprint(b)
    assert canonicalize(b)["a"]["y"] == [1, 2]


def test_legacy_record_is_strict_and_healed_on_safe_replay(client):
    """升级前写入的旧记录（request_hash 存 test_no、无快照）：改内容仍拒绝，安全重试后补严格化。"""
    with transaction(immediate=True) as connection:
        service = GermplasmService(connection)
        _, lot, _ = create_stored_lot(service, "110")
        protocol = _protocol(service)
        first = service.viability.schedule_test(_schedule_payload(lot, protocol))
        connection.execute(
            "UPDATE idempotency_records SET request_hash=?, request_json='{}', resource_json='{}' "
            "WHERE scope=? AND idempotency_key=?",
            ("VT-IDEM-001", SCHEDULE_SCOPE, "schedule-idem-0001"),
        )

        with pytest.raises(ConflictError) as exc_info:
            service.viability.schedule_test(_schedule_payload(lot, protocol, sampled_grams=7))
        assert exc_info.value.context["original_resource"]["test_id"] == first["id"]
        assert [c["field"] for c in exc_info.value.context["conflicts"]] == ["sampled_grams"]

        replay = service.viability.schedule_test(_schedule_payload(lot, protocol))
        assert replay["id"] == first["id"]
        healed = IdempotencyService(connection).get_record(SCHEDULE_SCOPE, "schedule-idem-0001")
        assert healed.request_snapshot["sampled_grams"] == 5
        assert healed.request_hash != "VT-IDEM-001"
        assert int(healed.resource["test_id"]) == first["id"]
        assert _counts(connection) == (1, 1, 0)
