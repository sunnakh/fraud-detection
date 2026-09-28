from datetime import UTC, datetime

import pytest

from fraud_detection.outcomes import (
    AmbiguousOutcomeHistoryError,
    OutcomeRecord,
    OutcomeStatus,
    outcome_as_of,
)


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def outcome(
    record_id: str,
    status: OutcomeStatus,
    available_day: int,
    *,
    supersedes_id: str | None = None,
) -> OutcomeRecord:
    return OutcomeRecord(
        record_id=record_id,
        transaction_id="tx-1",
        status=status,
        event_time=instant(1),
        available_at=instant(available_day),
        evidence_reference=f"evidence:{record_id}",
        supersedes_id=supersedes_id,
    )


def test_missing_outcome_remains_unknown() -> None:
    assert outcome_as_of((), "tx-1", instant(10)) is None


def test_future_outcome_does_not_leak_into_earlier_cutoff() -> None:
    fraud = outcome("label-v1", OutcomeStatus.FRAUD, available_day=10)

    assert outcome_as_of((fraud,), "tx-1", instant(9)) is None
    assert outcome_as_of((fraud,), "tx-1", instant(10)) == fraud


def test_correction_applies_only_after_it_is_available() -> None:
    initial = outcome("label-v1", OutcomeStatus.FRAUD, available_day=10)
    correction = outcome(
        "label-v2",
        OutcomeStatus.LEGITIMATE,
        available_day=20,
        supersedes_id="label-v1",
    )

    assert outcome_as_of((initial, correction), "tx-1", instant(15)) == initial
    assert outcome_as_of((initial, correction), "tx-1", instant(20)) == correction
    assert correction.effective_at == instant(1)


def test_branched_corrections_fail_explicitly() -> None:
    initial = outcome("label-v1", OutcomeStatus.FRAUD, available_day=10)
    correction_a = outcome(
        "label-v2a",
        OutcomeStatus.LEGITIMATE,
        available_day=20,
        supersedes_id="label-v1",
    )
    correction_b = outcome(
        "label-v2b",
        OutcomeStatus.DISPUTED,
        available_day=21,
        supersedes_id="label-v1",
    )

    with pytest.raises(AmbiguousOutcomeHistoryError, match="branched"):
        outcome_as_of((initial, correction_a, correction_b), "tx-1", instant(30))
