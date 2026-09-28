"""Pinned public-source loading and source-frame validation."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.simulation import SIMULATED_TIMEZONE

REQUIRED_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
)


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourcePins:
    repository: str
    commit: str
    files: tuple[SourceFile, ...]


def read_pins(path: Path) -> SourcePins:
    raw = json.loads(path.read_text())
    files = tuple(SourceFile(f["name"], f["sha256"]) for f in raw["files"])
    return SourcePins(raw["repository"], raw["commit"], files)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviewed_files(pins: SourcePins, raw_dir: Path) -> pd.DataFrame:
    """Verify every checksum before any pickle is deserialized."""

    paths = []
    for pinned in pins.files:
        path = raw_dir / pinned.name
        if sha256_file(path) != pinned.sha256:
            raise ValueError(f"checksum mismatch for {pinned.name}")
        paths.append(path)
    frames: list[pd.DataFrame] = []
    for path in paths:
        loaded = pd.read_pickle(path)
        if not isinstance(loaded, pd.DataFrame):
            raise ValueError(f"{path.name} does not contain a DataFrame")
        frames.append(loaded)
    return pd.concat(frames, ignore_index=True)


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject unusable rows; return a copy with UTC times and string IDs."""

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    if frame["TRANSACTION_ID"].isna().any():
        raise ValueError("TRANSACTION_ID contains nulls")
    if frame["TRANSACTION_ID"].duplicated().any():
        raise ValueError("duplicate TRANSACTION_ID values")
    if not pd.api.types.is_datetime64_any_dtype(frame["TX_DATETIME"]):
        raise ValueError("TX_DATETIME must be a datetime column")
    if frame["TX_DATETIME"].dt.tz is not None:
        raise ValueError("TX_DATETIME must be timezone-naive in the Handbook source")
    amounts = frame["TX_AMOUNT"].to_numpy(dtype=float)
    # The simulator rounds some small redrawn amounts to 0.00; keep those rows
    # so customer history counts match the source.
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("TX_AMOUNT must be finite and non-negative")
    if not frame["TX_FRAUD"].isin([0, 1]).all():
        raise ValueError("TX_FRAUD must be binary 0/1")
    result = frame.copy()
    result["TX_DATETIME"] = result["TX_DATETIME"].dt.tz_localize(SIMULATED_TIMEZONE)
    result["TRANSACTION_ID"] = result["TRANSACTION_ID"].astype("int64").astype(str)
    result["CUSTOMER_ID"] = result["CUSTOMER_ID"].astype(str)
    return result
