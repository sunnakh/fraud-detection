import hashlib
import pickle
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data import (
    SourceFile,
    SourcePins,
    load_reviewed_files,
    validate_source_frame,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2],
            "TX_DATETIME": pd.to_datetime(
                ["2018-04-01 00:00:31", "2018-04-01 00:02:10"]
            ),
            "CUSTOMER_ID": [596, 4961],
            "TX_AMOUNT": [57.16, 81.51],
            "TX_FRAUD": [0, 1],
        }
    )


def test_naive_times_follow_simulation_convention() -> None:
    result = validate_source_frame(valid_frame())
    assert str(result["TX_DATETIME"].dt.tz) == "UTC"


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="TX_FRAUD"):
        validate_source_frame(valid_frame().drop(columns="TX_FRAUD"))


def test_duplicate_id_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[1, "TRANSACTION_ID"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_source_frame(frame)


def test_null_id_is_rejected() -> None:
    frame = valid_frame().astype({"TRANSACTION_ID": "float"})
    frame.loc[0, "TRANSACTION_ID"] = None
    with pytest.raises(ValueError, match="TRANSACTION_ID"):
        validate_source_frame(frame)


@pytest.mark.parametrize("amount", [-1.0, -0.01, float("nan"), float("inf")])
def test_negative_or_non_finite_amount_is_rejected(amount: float) -> None:
    frame = valid_frame()
    frame.loc[0, "TX_AMOUNT"] = amount
    with pytest.raises(ValueError, match="TX_AMOUNT"):
        validate_source_frame(frame)


def test_zero_amount_from_simulator_rounding_is_kept() -> None:
    frame = valid_frame()
    frame.loc[0, "TX_AMOUNT"] = 0.0
    assert len(validate_source_frame(frame)) == 2


def test_non_binary_label_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[0, "TX_FRAUD"] = 2
    with pytest.raises(ValueError, match="TX_FRAUD"):
        validate_source_frame(frame)


def test_already_aware_times_are_rejected() -> None:
    frame = valid_frame()
    frame["TX_DATETIME"] = frame["TX_DATETIME"].dt.tz_localize("Europe/Paris")
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_source_frame(frame)


def test_checksum_mismatch_blocks_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "2018-04-01.pkl"
    valid_frame().to_pickle(path)
    pins = SourcePins("repo", "abc123", (SourceFile(path.name, "0" * 64),))
    with pytest.raises(ValueError, match="checksum"):
        load_reviewed_files(pins, tmp_path)


def test_pinned_file_must_hold_a_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "2018-04-01.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pins = SourcePins("repo", "abc123", (SourceFile(path.name, digest),))
    with pytest.raises(ValueError, match="DataFrame"):
        load_reviewed_files(pins, tmp_path)


def test_pinned_file_loads(tmp_path: Path) -> None:
    path = tmp_path / "2018-04-01.pkl"
    valid_frame().to_pickle(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pins = SourcePins("repo", "abc123", (SourceFile(path.name, digest),))
    assert len(load_reviewed_files(pins, tmp_path)) == 2
