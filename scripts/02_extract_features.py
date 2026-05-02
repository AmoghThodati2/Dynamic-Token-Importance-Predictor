"""
scripts/02_extract_features.py

Extract per-token features from all collected attention traces and write
data/features/features.parquet.

Each row in the output corresponds to one token position in one sample. The feature
columns are defined in token_importance.features.FEATURE_COLS. The file is ready for
label joining (labels.py) and XGBoost/MLP training in subsequent pipeline steps.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-token features from attention traces."
    )
    p.add_argument("--trace-dir", type=Path, default=Path("data/traces"))
    p.add_argument(
        "--output-path", type=Path, default=Path("data/features/features.parquet")
    )
    p.add_argument(
        "--recent-window",
        type=int,
        default=8,
        help="Threshold for is_recent: 1 if pos_from_end < recent_window, else 0.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N traces (useful for dry runs).",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
        force=True,
    )

    from token_importance.features import FEATURE_COLS, build_feature_table

    table = build_feature_table(
        trace_dir=args.trace_dir,
        output_path=args.output_path,
        recent_window=args.recent_window,
        max_samples=args.max_samples,
    )

    print(f"\n{len(table):,} rows  ×  {len(table.columns)} columns")
    print(f"Features: {FEATURE_COLS}")
    print("\nPer-column stats:")
    print(table[FEATURE_COLS].describe().T.to_string())


if __name__ == "__main__":
    main()
