"""Versioned delayed-outcome handling without negative-label inference."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from fraud_detection.temporal import TemporalRecord, ensure_utc, known_as_of


class OutcomeStatus(StrEnum):
    """Candidate outcome states; evidence rules remain stakeholder-owned."""

    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    DISPUTED = "disputed"


@dataclass(frozen=True, slots=True)
class OutcomeRecord(TemporalRecord):
    """An immutable outcome assertion or correction.

    Absence of a record represents an unknown outcome. ``supersedes_id`` links
    a correction to the immutable assertion it replaces.
    """

    transaction_id: str
    status: OutcomeStatus
    evidence_reference: str
    supersedes_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.transaction_id.strip():
            raise ValueError("transaction_id must not be blank")
        if not self.evidence_reference.strip():
            raise ValueError("evidence_reference must not be blank")
        if self.supersedes_id == self.record_id:
            raise ValueError("an outcome cannot supersede itself")

    @property
    def effective_at(self) -> datetime:
        """Return when this outcome assertion applies in event time."""

        return self.event_time


class AmbiguousOutcomeHistoryError(ValueError):
    """Raised when an outcome history cannot yield one authoritative version."""


def outcome_as_of(
    records: Iterable[OutcomeRecord],
    transaction_id: str,
    cutoff: datetime,
) -> OutcomeRecord | None:
    """Return the unsuperseded outcome known at ``cutoff``.

    ``None`` means unknown at that cutoff. It never means legitimate. Corrections
    become effective for knowledge-time evaluation only once they are available.
    Branched corrections fail explicitly instead of choosing one silently.
    """

    normalized_cutoff = ensure_utc(cutoff, field_name="cutoff")
    relevant = tuple(
        record
        for record in records
        if record.transaction_id == transaction_id
        and record.available_at <= normalized_cutoff
    )
    if not relevant:
        return None

    by_id: dict[str, OutcomeRecord] = {}
    for record in relevant:
        if record.record_id in by_id:
            message = f"duplicate outcome record_id: {record.record_id}"
            raise AmbiguousOutcomeHistoryError(message)
        by_id[record.record_id] = record

    superseded_ids: set[str] = set()
    for record in known_as_of(relevant, normalized_cutoff):
        if record.supersedes_id is None:
            continue
        predecessor = by_id.get(record.supersedes_id)
        if predecessor is None:
            message = (
                f"outcome {record.record_id} supersedes unavailable record "
                f"{record.supersedes_id}"
            )
            raise AmbiguousOutcomeHistoryError(message)
        if predecessor.transaction_id != record.transaction_id:
            message = "an outcome cannot supersede another transaction's outcome"
            raise AmbiguousOutcomeHistoryError(message)
        if record.supersedes_id in superseded_ids:
            message = f"branched correction history at {record.supersedes_id}"
            raise AmbiguousOutcomeHistoryError(message)
        superseded_ids.add(record.supersedes_id)

    active = tuple(
        record for record in relevant if record.record_id not in superseded_ids
    )
    if len(active) != 1:
        message = (
            f"expected one active outcome for {transaction_id}, found {len(active)}"
        )
        raise AmbiguousOutcomeHistoryError(message)
    return active[0]
