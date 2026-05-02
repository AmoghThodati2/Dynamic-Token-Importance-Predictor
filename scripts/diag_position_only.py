"""
scripts/diag_position_only.py

Decisive diagnostic for the predictor's training/inference feature mismatch.

At Stage 2 inference time, the kelle-simulator's `_head_attention_weights()` callsite
only has position-based information (sorted token positions, layer index, head index,
recent-window/initial-tokens hardware config). None of the attention-statistics features
in the current FEATURE_COLS (cum_attn, last_attn, head_agree_*, etc.) are computable
there — they require the full prefill (T, T) attention matrix.

This script trains XGBoost twice on the existing dataset and reports test AUC for both:
  (a) FULL feature set — what the current pipeline produces
  (b) POSITION-ONLY feature set — what's actually available at inference

The gap between the two AUC numbers is the load-bearing signal:
  gap small         → MLP can be trained on inference-realistic features. Proceed.
  gap large         → Either redefine labels, add evictor-state features, or pivot
                       to a different problem formulation. Do not write the MLP yet.

Decision thresholds (informal, calibrate on real data):
  position-only AUC >= 0.75 → proceed to MLP
  position-only AUC in [0.6, 0.75] → marginal; try evictor-state features
  position-only AUC < 0.6 → label cannot be predicted from position alone
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POSITION_ONLY_FEATURES = [
    "pos",
    "rel_pos",
    "recency",
    "is_sink",
    "is_recent",  # derived in this script: pos_from_end < recent_window
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare full-feature vs position-only XGBoost AUC."
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/features/features.parquet")
    )
    p.add_argument(
        "--labels-path", type=Path, default=Path("data/features/labels.parquet")
    )
    p.add_argument(
        "--recent-window",
        type=int,
        default=8,
        help="Tokens within this distance from end are flagged is_recent=1.",
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def add_inference_features(df: pd.DataFrame, recent_window: int) -> pd.DataFrame:
    """Derive position-based columns that mirror what the simulator can compute."""
    seq_len_per_sample = df.groupby("sample_id")["pos"].transform("max") + 1
    df = df.copy()
    df["pos_from_end"] = seq_len_per_sample - 1 - df["pos"]
    df["is_recent"] = (df["pos_from_end"] < recent_window).astype(np.float32)
    return df


def run_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    seed: int,
) -> dict:
    """Train + evaluate one XGBoost configuration; return a metrics dict."""
    import token_importance.models.xgb as xgb_mod

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32) if len(val_df) else None
    y_val = val_df["label"].to_numpy() if len(val_df) else None
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32) if len(test_df) else None
    y_test = test_df["label"].to_numpy() if len(test_df) else None

    model = xgb_mod.train(
        x_train, y_train, feature_cols,
        x_val=x_val, y_val=y_val,
        n_estimators=200, max_depth=4, learning_rate=0.1,
        early_stopping_rounds=20, seed=seed,
    )

    out = {"label": label, "n_features": len(feature_cols), "features": feature_cols}
    for split_name, x_arr, y_arr in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        if x_arr is None or len(x_arr) == 0:
            continue
        m = xgb_mod.evaluate(model, x_arr, y_arr, feature_cols)
        out[f"{split_name}_auc"] = m.get("roc_auc", float("nan"))
        out[f"{split_name}_acc"] = m.get("accuracy", float("nan"))
    return out


def interpret(pos_auc: float, full_auc: float) -> str:
    if np.isnan(pos_auc):
        return "INCONCLUSIVE — single-class test split (need more samples)."
    gap = full_auc - pos_auc
    if pos_auc >= 0.75:
        return f"PROCEED — position-only AUC={pos_auc:.3f} is strong; gap to full={gap:.3f}."
    if pos_auc >= 0.60:
        return (
            f"MARGINAL — position-only AUC={pos_auc:.3f}. Try evictor-state features "
            f"(Step 3a) before writing the MLP."
        )
    return (
        f"BLOCKED — position-only AUC={pos_auc:.3f} is at chance. The label cannot be "
        f"predicted from position alone. Consider Step 3b (forward-looking labels) or "
        f"pivot to per-(layer,head) attention-distribution prediction."
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
        force=True,
    )

    import token_importance.models.xgb as xgb_mod
    from token_importance.features import FEATURE_COLS

    logging.info("Loading dataset")
    df = xgb_mod.load_dataset(args.features_path, args.labels_path)
    df = add_inference_features(df, args.recent_window)

    train_df, val_df, test_df = xgb_mod.split_by_sample(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    logging.info(
        "Split: %d / %d / %d rows (train/val/test); positive rate: train=%.1f%% test=%.1f%%",
        len(train_df), len(val_df), len(test_df),
        100.0 * train_df["label"].mean() if len(train_df) else float("nan"),
        100.0 * test_df["label"].mean() if len(test_df) else float("nan"),
    )

    full_result = run_xgb(train_df, val_df, test_df, FEATURE_COLS, "FULL", args.seed)
    pos_result = run_xgb(
        train_df, val_df, test_df, POSITION_ONLY_FEATURES, "POSITION-ONLY", args.seed
    )

    print("\n" + "=" * 70)
    print("Diagnostic: full feature set vs position-only feature set")
    print("=" * 70)
    rows = []
    for r in (full_result, pos_result):
        rows.append({
            "label": r["label"],
            "n_features": r["n_features"],
            "train_auc": r.get("train_auc"),
            "val_auc": r.get("val_auc"),
            "test_auc": r.get("test_auc"),
            "test_acc": r.get("test_acc"),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nPosition-only feature set: {POSITION_ONLY_FEATURES}")
    print(f"Full feature set:          {FEATURE_COLS}")

    pos_auc = pos_result.get("test_auc", float("nan"))
    full_auc = full_result.get("test_auc", float("nan"))
    print(f"\nVerdict: {interpret(pos_auc, full_auc)}")


if __name__ == "__main__":
    main()
