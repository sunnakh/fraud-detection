"""Command-line entry point for offline and operational jobs."""

import argparse
import json
import urllib.request
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

from fraud_detection.data import sha256_file

HANDBOOK_REPO = "Fraud-Detection-Handbook/simulated-data-transformed"


def _fetch_source(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    day = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    while day <= end:
        name = f"{day.isoformat()}.pkl"
        url = (
            f"https://raw.githubusercontent.com/{HANDBOOK_REPO}/{args.commit}"
            f"/data/{name}"
        )
        target = raw_dir / name
        urllib.request.urlretrieve(url, target)
        files.append({"name": name, "sha256": sha256_file(target)})
        day += timedelta(days=1)
    pins = {"repository": HANDBOOK_REPO, "commit": args.commit, "files": files}
    pins_path = Path(args.pins)
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"pinned {len(files)} files at {args.commit}; review before loading")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-detection")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-source", help="download and pin Handbook files")
    fetch.add_argument("--commit", required=True)
    fetch.add_argument("--start", default="2018-04-01")
    fetch.add_argument("--end", default="2018-09-30")
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--pins", default="config/source-pins.json")
    fetch.set_defaults(handler=_fetch_source)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
