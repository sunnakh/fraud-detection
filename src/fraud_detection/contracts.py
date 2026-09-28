"""Draft, source-neutral artifact contracts for reproducible evaluation."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite

from fraud_detection.temporal import ensure_utc


class ScoreStatus(StrEnum):
    """Whether a score was produced; downstream action remains external."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ScoreKind(StrEnum):
    """Original results are immutable; replays are separate later records."""

    ORIGINAL = "original"
    REPLAY = "replay"


class UnavailableReason(StrEnum):
    """Machine-readable reasons a valid transaction was not scored."""

    LATE = "late"
    FEATURE_FAILURE = "feature_failure"
    MODEL_UNAVAILABLE = "model_unavailable"


class HistoryStatus(StrEnum):
    """Explicit cold start instead of an implicit zero history."""

    NO_HISTORY = "no_history"
    HAS_HISTORY = "has_history"


RANKING_SEMANTICS = "ranking"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Identity and time boundary for a reproducible logical dataset."""

    dataset_version: str
    created_at: datetime
    knowledge_cutoff: datetime
    source_references: tuple[str, ...]
    record_count: int
    content_checksum: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("dataset_version", self.dataset_version),
            ("content_checksum", self.content_checksum),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if not self.source_references or any(
            not reference.strip() for reference in self.source_references
        ):
            raise ValueError("source_references must contain nonblank references")
        if self.record_count < 0:
            raise ValueError("record_count must be nonnegative")
        object.__setattr__(
            self,
            "created_at",
            ensure_utc(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "knowledge_cutoff",
            ensure_utc(self.knowledge_cutoff, field_name="knowledge_cutoff"),
        )
        if self.created_at < self.knowledge_cutoff:
            raise ValueError("created_at must not precede knowledge_cutoff")


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Traceable scoring result; never carries a payment action."""

    transaction_id: str
    scored_at: datetime
    status: ScoreStatus
    score: float | None
    score_semantics: str | None
    model_version: str | None
    feature_version: str | None
    reason: UnavailableReason | None = None
    kind: ScoreKind = ScoreKind.ORIGINAL

    def __post_init__(self) -> None:
        if not self.transaction_id.strip():
            raise ValueError("transaction_id must not be blank")
        for field_name, value in (
            ("model_version", self.model_version),
            ("feature_version", self.feature_version),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        object.__setattr__(
            self,
            "scored_at",
            ensure_utc(self.scored_at, field_name="scored_at"),
        )
        if self.status is ScoreStatus.AVAILABLE:
            if self.score is None or self.score_semantics is None:
                raise ValueError(
                    "available scores require a value and explicit semantics"
                )
            if not isfinite(self.score):
                raise ValueError("available scores must be finite")
            if not self.score_semantics.strip():
                raise ValueError("score_semantics must not be blank")
            if self.reason is not None:
                raise ValueError("available scores must not include a failure reason")
            if self.model_version is None:
                raise ValueError("available scores require a model_version")
            if self.feature_version is None:
                raise ValueError("available scores require a feature_version")
        elif self.score is not None or self.score_semantics is not None:
            raise ValueError("unavailable results must not manufacture a score")
        elif self.reason is None:
            raise ValueError("unavailable results require a reason")
        else:
            try:
                reason = UnavailableReason(self.reason)
            except ValueError as exc:
                allowed = ", ".join(r.value for r in UnavailableReason)
                raise ValueError(f"reason must be one of: {allowed}") from exc
            object.__setattr__(self, "reason", reason)
