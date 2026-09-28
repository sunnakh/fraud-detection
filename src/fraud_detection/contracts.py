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
    """Traceable scoring result with deliberately uncommitted semantics."""

    transaction_id: str
    scored_at: datetime
    status: ScoreStatus
    score: float | None
    score_semantics: str | None
    model_version: str
    feature_version: str
    reason: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("transaction_id", self.transaction_id),
            ("model_version", self.model_version),
            ("feature_version", self.feature_version),
        ):
            if not value.strip():
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
        elif self.score is not None or self.score_semantics is not None:
            raise ValueError("unavailable results must not manufacture a score")
        elif self.reason is None or not self.reason.strip():
            raise ValueError("unavailable results require a reason")
