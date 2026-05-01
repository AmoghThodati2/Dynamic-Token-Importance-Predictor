"""
scripts/04_train_xgb.py

Train an XGBClassifier on the joined (features, labels) dataset, rank features by gain
importance, and write:
  data/models/xgb_selector.json     — serialized XGBoost model
  data/models/selected_features.json — ordered list of top-K feature names for the MLP

Feature selection output is the only artifact the MLP training step consumes from this
script. The XGBoost model itself is retained as a classification baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost for feature ranking and baseline classification."
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/features/features.parquet")
    )
    p.add_argument(
        "--labels-path", type=Path, default=Path("data/features/labels.parquet")
    )
    p.add_argument(
        "--model-path", type=Path, default=Path("data/models/xgb_selector.json")
    )
    p.add_argument(
        "--features-out",
        type=Path,
        default=Path("data/models/selected_features.json"),
        help="Path to write the ordered list of top-K selected feature names.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top features (by gain) to select for MLP input.",
    )
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
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

    import token_importance.models.xgb as xgb_mod
    from token_importance.features import FEATURE_COLS

    # ---- load & split ----
    logging.info("Loading dataset from %s + %s", args.features_path, args.labels_path)
    df = xgb_mod.load_dataset(args.features_path, args.labels_path)
    logging.info("Dataset: %d rows, %.1f%% positive", len(df), 100.0 * df["label"].mean())

    train_df, val_df, test_df = xgb_mod.split_by_sample(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    logging.info(
        "Split by sample_id: %d / %d / %d rows (train/val/test)",
        len(train_df), len(val_df), len(test_df),
    )

    x_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    x_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32) if len(val_df) else None
    y_val = val_df["label"].to_numpy() if len(val_df) else None
    x_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32) if len(test_df) else None
    y_test = test_df["label"].to_numpy() if len(test_df) else None

    # ---- train ----
    logging.info("Training XGBClassifier (%d estimators max)...", args.n_estimators)
    model = xgb_mod.train(
        x_train, y_train, FEATURE_COLS,
        x_val=x_val, y_val=y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )

    # ---- feature importances ----
    imps = xgb_mod.feature_importances(model, FEATURE_COLS)
    print("\nFeature importances (sorted by gain):")
    print(imps.to_string(index=False))

    top_k = min(args.top_k, len(imps))
    top_features = imps.head(top_k)["feature"].tolist()
    print(f"\nTop-{top_k} features selected for MLP: {top_features}")

    # ---- evaluate ----
    for split_name, x_arr, y_arr in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        if x_arr is None or len(x_arr) == 0:
            continue
        metrics = xgb_mod.evaluate(model, x_arr, y_arr, FEATURE_COLS)
        parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"\n{split_name:5s}: {parts}")

    # ---- save ----
    xgb_mod.save_model(model, args.model_path)

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.write_text(json.dumps(top_features, indent=2))
    logging.info("Saved selected features → %s", args.features_out)

    print("\nArtifacts written:")
    print(f"  {args.model_path}")
    print(f"  {args.features_out}")


if __name__ == "__main__":
    main()
