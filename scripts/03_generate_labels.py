"""
scripts/03_generate_labels.py

Generate per-token AERP binary labels from attention traces and write
data/features/labels.parquet.

Each row aligns with a row in data/features/features.parquet on (sample_id, token_pos).
Join the two files to get the full (X, y) dataset for XGBoost / MLP training.

Label rule: token k is labeled 1 if ≥ vote_threshold fraction of (layer, head) pairs
rank it among the top --cache-size tokens by cumulative attention received.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate AERP token-importance labels from attention traces."
    )
    p.add_argument("--trace-dir", type=Path, default=Path("data/traces"))
    p.add_argument(
        "--output-path", type=Path, default=Path("data/features/labels.parquet")
    )
    p.add_argument(
        "--cache-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "KV cache capacity: tokens each head retains (default: seq_len // 2). "
            "Use 128 to match the kelle-simulator reference setting when seq_len > 128."
        ),
    )
    p.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of (layer, head) pairs voting to retain for label=1.",
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

    import pandas as pd

    from token_importance.labels import build_label_table

    table = build_label_table(
        trace_dir=args.trace_dir,
        output_path=args.output_path,
        cache_size=args.cache_size,
        vote_threshold=args.vote_threshold,
        max_samples=args.max_samples,
    )

    print(f"\n{len(table):,} rows — label distribution:")
    print(table["label"].value_counts().to_string())
    print("\nvote_frac stats:")
    print(table["vote_frac"].describe().to_string())

    # Show the joined shape as a quick sanity check
    feat_path = args.trace_dir.parent / "features" / "features.parquet"
    if feat_path.exists():
        features = pd.read_parquet(feat_path)
        joined = features.merge(table, on=["sample_id", "token_pos"], how="inner")
        print(f"\nJoined with feature table: {len(joined):,} rows × {len(joined.columns)} cols")
        if len(joined) != len(table):
            print(
                f"WARNING: join dropped {len(table) - len(joined)} rows "
                f"— re-run 02_extract_features.py if traces changed"
            )


if __name__ == "__main__":
    main()
