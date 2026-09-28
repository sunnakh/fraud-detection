"""Transaction fraud risk scoring package.

The package currently provides source-neutral correctness primitives. It does
not yet contain a bank-approved schema, trained model, or scoring service.
"""

from fraud_detection.contracts import DatasetManifest, ScoreResult, ScoreStatus
from fraud_detection.outcomes import (
    AmbiguousOutcomeHistoryError,
    OutcomeRecord,
    OutcomeStatus,
    outcome_as_of,
)
from fraud_detection.temporal import (
    TemporalRecord,
    ensure_utc,
    known_as_of,
    point_in_time_records,
)

__all__ = [
    "AmbiguousOutcomeHistoryError",
    "DatasetManifest",
    "OutcomeRecord",
    "OutcomeStatus",
    "ScoreResult",
    "ScoreStatus",
    "TemporalRecord",
    "ensure_utc",
    "known_as_of",
    "outcome_as_of",
    "point_in_time_records",
]
