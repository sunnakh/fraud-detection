from datetime import UTC, datetime, timedelta, timezone

import pytest

from fraud_detection.temporal import (
    TemporalRecord,
    ensure_utc,
    known_as_of,
    point_in_time_records,
)


def instant(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, tzinfo=UTC)


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="must include a timezone"):
        ensure_utc(datetime(2026, 1, 1), field_name="event_time")


def test_timestamps_are_normalized_to_utc() -> None:
    offset = timezone(timedelta(hours=5))

    normalized = ensure_utc(datetime(2026, 1, 1, 15, tzinfo=offset))

    assert normalized == instant(10)
    assert normalized.tzinfo is UTC


def test_future_arrival_is_not_known_at_decision_time() -> None:
    on_time = TemporalRecord("known", event_time=instant(8), available_at=instant(9))
    late = TemporalRecord("late", event_time=instant(7), available_at=instant(11))

    assert known_as_of((late, on_time), instant(10)) == (on_time,)


def test_exact_cutoff_behavior_is_explicit() -> None:
    boundary = TemporalRecord(
        "boundary", event_time=instant(8), available_at=instant(10)
    )

    assert known_as_of((boundary,), instant(10), inclusive=True) == (boundary,)
    assert known_as_of((boundary,), instant(10), inclusive=False) == ()


def test_point_in_time_filter_requires_event_and_availability_eligibility() -> None:
    eligible = TemporalRecord(
        "eligible", event_time=instant(8), available_at=instant(9)
    )
    future_event = TemporalRecord(
        "future-event", event_time=instant(11), available_at=instant(9)
    )
    late_arrival = TemporalRecord(
        "late-arrival", event_time=instant(8), available_at=instant(11)
    )

    result = point_in_time_records(
        (future_event, late_arrival, eligible),
        event_cutoff=instant(10),
        knowledge_cutoff=instant(10),
    )

    assert result == (eligible,)


def test_point_in_time_boundaries_are_independently_configurable() -> None:
    boundary = TemporalRecord(
        "boundary", event_time=instant(10), available_at=instant(10)
    )

    assert point_in_time_records(
        (boundary,),
        event_cutoff=instant(10),
        knowledge_cutoff=instant(10),
    ) == (boundary,)
    assert (
        point_in_time_records(
            (boundary,),
            event_cutoff=instant(10),
            knowledge_cutoff=instant(10),
            include_event_cutoff=False,
        )
        == ()
    )
    assert (
        point_in_time_records(
            (boundary,),
            event_cutoff=instant(10),
            knowledge_cutoff=instant(10),
            include_knowledge_cutoff=False,
        )
        == ()
    )
