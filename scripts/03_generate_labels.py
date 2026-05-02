"""
scripts/03_generate_labels.py

Generate per-(layer, head, token_pos) above-mean binary labels from attention traces
and write data/features/labels.parquet.

Each row in the output is uniquely keyed by (sample_id, token_pos, layer, head) and
aligns row-for-row with features.parquet produced by 02_extract_features.py.

Label rule: label = 1 if attn[l, h, T-1, k] > mean(attn[l, h, T-1, :]) else 0 — i.e.,
'did this head's last-query attention to this token exceed the head's mean over all
keys?'. This is the per-head increment condition used by the simulator's
systolic_evictor.accumulate_scores. See src/token_importance/labels.py for full
discussion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-(layer, head, token) above-mean labels."
    )
    p.add_argument("--trace-dir", type=Path, default=Path("data/traces"))
    p.add_argument(
        "--output-path", type=Path, default=Path("data/features/labels.parquet")
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
        max_samples=args.max_samples,
    )

    print(f"\n{len(table):,} rows — label distribution:")
    print(table["label"].value_counts().to_string())
    print(f"\npositive rate: {table['label'].mean():.4f}")

    feat_path = args.trace_dir.parent / "features" / "features.parquet"
    if feat_path.exists():
        features = pd.read_parquet(feat_path)
        joined = features.merge(
            table, on=["sample_id", "token_pos", "layer", "head"], how="inner"
        )
        print(f"\nJoined with feature table: {len(joined):,} rows × {len(joined.columns)} cols")
        if len(joined) != len(table):
            print(
                f"WARNING: join dropped {len(table) - len(joined)} rows "
                f"— re-run 02_extract_features.py if traces changed"
            )


if __name__ == "__main__":
    main()
