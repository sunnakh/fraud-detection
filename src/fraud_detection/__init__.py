"""Transaction fraud risk scoring package.

The package currently provides source-neutral correctness primitives. It does
not yet contain a bank-approved schema, trained model, or scoring service.
"""

from fraud_detection.contracts import (
    RANKING_SEMANTICS,
    DatasetManifest,
    HistoryStatus,
    ScoreKind,
    ScoreResult,
    ScoreStatus,
    UnavailableReason,
)
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
    "HistoryStatus",
    "OutcomeRecord",
    "OutcomeStatus",
    "RANKING_SEMANTICS",
    "ScoreKind",
    "ScoreResult",
    "ScoreStatus",
    "TemporalRecord",
    "UnavailableReason",
    "ensure_utc",
    "known_as_of",
    "outcome_as_of",
    "point_in_time_records",
]
