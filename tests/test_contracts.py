from datetime import UTC, datetime

import pytest

from fraud_detection.contracts import DatasetManifest, ScoreResult, ScoreStatus


def instant() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_dataset_manifest_rejects_negative_record_count() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        DatasetManifest(
            dataset_version="dataset-v1",
            created_at=instant(),
            knowledge_cutoff=instant(),
            source_references=("approved-source:pending",),
            record_count=-1,
            content_checksum="sha256:example",
        )


def test_dataset_manifest_rejects_creation_before_knowledge_cutoff() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        DatasetManifest(
            dataset_version="dataset-v1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            knowledge_cutoff=datetime(2026, 1, 2, tzinfo=UTC),
            source_references=("approved-source:pending",),
            record_count=0,
            content_checksum="sha256:example",
        )


def test_unavailable_result_cannot_contain_a_score() -> None:
    with pytest.raises(ValueError, match="must not manufacture"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.UNAVAILABLE,
            score=0.75,
            score_semantics="unapproved-ranking-value",
            model_version="model-v1",
            feature_version="features-v1",
            reason="model unavailable",
        )


def test_unavailable_result_requires_an_explicit_reason() -> None:
    with pytest.raises(ValueError, match="require a reason"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.UNAVAILABLE,
            score=None,
            score_semantics=None,
            model_version="model-v1",
            feature_version="features-v1",
        )


def test_available_result_is_traceable_without_owning_policy() -> None:
    result = ScoreResult(
        transaction_id="tx-1",
        scored_at=instant(),
        status=ScoreStatus.AVAILABLE,
        score=0.75,
        score_semantics="candidate-ranking-value",
        model_version="model-v1",
        feature_version="features-v1",
    )

    assert result.score == 0.75
    assert not hasattr(result, "action")


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_available_result_requires_a_finite_score(invalid_score: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.AVAILABLE,
            score=invalid_score,
            score_semantics="candidate-ranking-value",
            model_version="model-v1",
            feature_version="features-v1",
        )


def test_available_result_requires_nonblank_semantics() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        ScoreResult(
            transaction_id="tx-1",
            scored_at=instant(),
            status=ScoreStatus.AVAILABLE,
            score=0.5,
            score_semantics="  ",
            model_version="model-v1",
            feature_version="features-v1",
        )
