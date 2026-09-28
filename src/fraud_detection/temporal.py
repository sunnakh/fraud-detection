"""Source-neutral point-in-time correctness primitives.

These types deliberately model when an event happened separately from when it
became knowable. They do not prescribe a bank transaction schema or storage
technology.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime


def ensure_utc(value: datetime, *, field_name: str = "timestamp") -> datetime:
    """Return an aware timestamp normalized to UTC.

    Naive timestamps are rejected because silently guessing a timezone can move
    an event across a feature or label cutoff.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        message = f"{field_name} must include a timezone"
        raise ValueError(message)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class TemporalRecord:
    """A source-neutral observation with event and availability times."""

    record_id: str
    event_time: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must not be blank")
        object.__setattr__(
            self,
            "event_time",
            ensure_utc(self.event_time, field_name="event_time"),
        )
        object.__setattr__(
            self,
            "available_at",
            ensure_utc(self.available_at, field_name="available_at"),
        )


def known_as_of[RecordT: TemporalRecord](
    records: Iterable[RecordT],
    cutoff: datetime,
    *,
    inclusive: bool = True,
) -> tuple[RecordT, ...]:
    """Return records that were knowable at ``cutoff`` in stable time order.

    ``inclusive`` is explicit because whether an observation arriving exactly
    at the decision instant is usable depends on the eventual bank contract.
    """

    normalized_cutoff = ensure_utc(cutoff, field_name="cutoff")

    if inclusive:
        eligible = (
            record for record in records if record.available_at <= normalized_cutoff
        )
    else:
        eligible = (
            record for record in records if record.available_at < normalized_cutoff
        )

    return tuple(
        sorted(
            eligible,
            key=lambda record: (
                record.available_at,
                record.event_time,
                record.record_id,
            ),
        )
    )


def point_in_time_records[RecordT: TemporalRecord](
    records: Iterable[RecordT],
    *,
    event_cutoff: datetime,
    knowledge_cutoff: datetime,
    include_event_cutoff: bool = True,
    include_knowledge_cutoff: bool = True,
) -> tuple[RecordT, ...]:
    """Return records satisfying independent event- and knowledge-time cutoffs.

    A record must both have happened within the event window and have arrived by
    the decision's knowledge boundary. Separate inclusion switches keep exact
    boundary semantics visible until the bank contract approves them.
    """

    normalized_event_cutoff = ensure_utc(event_cutoff, field_name="event_cutoff")
    known_records = known_as_of(
        records,
        knowledge_cutoff,
        inclusive=include_knowledge_cutoff,
    )
    if include_event_cutoff:
        return tuple(
            record
            for record in known_records
            if record.event_time <= normalized_event_cutoff
        )
    return tuple(
        record
        for record in known_records
        if record.event_time < normalized_event_cutoff
    )
