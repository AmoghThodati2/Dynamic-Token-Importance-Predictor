"""
scripts/05_train_mlp.py

Train the MLP token-importance predictor on the 7 position-only features and evaluate
on the held-out test split.

Uses the identical sample-id-based train/val/test split as 04_train_xgb.py (seed=42,
val_frac=0.15, test_frac=0.15) so MLP and XGBoost test AUCs are directly comparable.

Artifacts written to data/models/:
  mlp.pt              — model state dict (torch.save)
  scaler.pkl          — fitted StandardScaler (pickle, protocol 4)
  feature_schema.json — feature list, dimensions, and test metrics
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from token_importance.models import xgb as xgb_mod
from token_importance.models.mlp import FEATURE_COLS, Trainer

FEATURES_PATH = Path("data/features/features.parquet")
LABELS_PATH = Path("data/features/labels.parquet")
MODELS_DIR = Path("data/models")

XGB_REFERENCE_AUC = 0.8008  # 04_train_xgb.py test AUC under per-head 4-key join

TRAIN_SUBSAMPLE = 5_000_000  # cap training rows to keep CPU epochs tractable
BATCH_SIZE = 16384


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    # ---- load ----
    logging.info("Loading dataset from %s + %s", FEATURES_PATH, LABELS_PATH)
    df = xgb_mod.load_dataset(FEATURES_PATH, LABELS_PATH)
    logging.info("Dataset: %d rows, %.1f%% positive", len(df), 100.0 * df["label"].mean())

    # ---- split (same seed/ratios as 04_train_xgb.py) ----
    train_df, val_df, test_df = xgb_mod.split_by_sample(
        df, val_frac=0.15, test_frac=0.15, seed=42
    )
    logging.info(
        "Split by sample_id: %d / %d / %d rows (train/val/test)",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    if TRAIN_SUBSAMPLE is not None and len(train_df) > TRAIN_SUBSAMPLE:
        train_df = train_df.sample(n=TRAIN_SUBSAMPLE, random_state=42).reset_index(drop=True)
        logging.info("Subsampled train: %d rows (cap=%d)", len(train_df), TRAIN_SUBSAMPLE)

    x_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    x_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()

    # ---- train ----
    logging.info(
        "Training MLP (epochs=50, lr=1e-3, wd=1e-4, batch=%d, patience=7)...", BATCH_SIZE
    )
    result = Trainer(batch_size=BATCH_SIZE).fit(x_train, y_train, x_val, y_val)
    model = result["model"]
    scaler = result["scaler"]

    # ---- test evaluation ----
    x_test_s = scaler.transform(x_test).astype(np.float32)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(x_test_s))
    test_probs = torch.sigmoid(test_logits).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    test_auc = float(roc_auc_score(y_test, test_probs))
    test_acc = float(accuracy_score(y_test, test_preds))

    # ---- save ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlp_path = MODELS_DIR / "mlp.pt"
    scaler_path = MODELS_DIR / "scaler.pkl"
    schema_path = MODELS_DIR / "feature_schema.json"

    torch.save(model.state_dict(), mlp_path)
    logging.info("Saved model → %s", mlp_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=4)
    logging.info("Saved scaler → %s", scaler_path)

    schema: dict = {
        "feature_cols": FEATURE_COLS,
        "input_dim": len(FEATURE_COLS),
        "test_auc": round(test_auc, 6),
        "test_acc": round(test_acc, 6),
        "best_val_auc": round(result["best_val_auc"], 6),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
    }
    schema_path.write_text(json.dumps(schema, indent=2))
    logging.info("Saved schema → %s", schema_path)

    # ---- final summary ----
    print(
        f"\nMLP test AUC  = {test_auc:.4f}"
        f"  (XGBoost per-head reference: {XGB_REFERENCE_AUC:.4f})"
    )
    print(f"MLP test acc  = {test_acc:.3f}")
    print(f"Best val AUC  = {result['best_val_auc']:.3f}")
    print("Saved: data/models/{mlp.pt, scaler.pkl, feature_schema.json}")


if __name__ == "__main__":
    main()
