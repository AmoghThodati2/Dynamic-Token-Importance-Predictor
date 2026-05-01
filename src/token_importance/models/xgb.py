"""
XGBoost wrapper for feature importance ranking and baseline classification.

Trains an XGBClassifier on the full feature table, extracts per-feature gain/weight/cover
importances to identify the top-K features for MLP input, and serializes the fitted model
to data/models/xgb_selector.json. The model also serves as a standalone baseline:
if XGBoost's AUC is poor, the feature set or label definition should be revisited before
building the MLP.

Key design choices:
  - DataFrames are passed to fit() so the booster stores feature names — required for
    named importance scores and for reloading without a separate feature list.
  - Split is by sample_id (not by row) to prevent token-level data leakage between
    train and test splits of the same sequence.
  - Early stopping is enabled whenever a validation set is present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def load_dataset(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """Join feature and label Parquet files on (sample_id, token_pos)."""
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)[["sample_id", "token_pos", "label"]]
    merged = features.merge(labels, on=["sample_id", "token_pos"], how="inner")
    if len(merged) != len(features):
        logging.warning(
            "Join dropped %d rows — features and labels may be out of sync",
            len(features) - len(merged),
        )
    return merged


def split_by_sample(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test partitions by sample_id.

    Grouping by sample_id prevents tokens from the same sequence appearing in both
    train and test, which would leak sequence-level patterns into the evaluation.
    """
    rng = np.random.default_rng(seed)
    sample_ids = df["sample_id"].unique().copy()
    rng.shuffle(sample_ids)

    n = len(sample_ids)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_ids = set(sample_ids[:n_test])
    val_ids = set(sample_ids[n_test : n_test + n_val])
    train_ids = set(sample_ids[n_test + n_val :])

    if not train_ids:
        logging.warning(
            "No training samples after split (%d total) — reduce val/test fractions", n
        )

    return (
        df[df["sample_id"].isin(train_ids)].reset_index(drop=True),
        df[df["sample_id"].isin(val_ids)].reset_index(drop=True),
        df[df["sample_id"].isin(test_ids)].reset_index(drop=True),
    )


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 20,
    seed: int = 42,
) -> XGBClassifier:
    """Train an XGBClassifier with optional early stopping on a validation set."""
    has_val = x_val is not None and y_val is not None and len(x_val) > 0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds if has_val else None,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    x_tr = pd.DataFrame(x_train, columns=feature_names)
    fit_kwargs: dict = {"verbose": False}
    if has_val:
        fit_kwargs["eval_set"] = [(pd.DataFrame(x_val, columns=feature_names), y_val)]

    model.fit(x_tr, y_train, **fit_kwargs)
    logging.info(
        "XGBoost trained: n_estimators=%d, best_iteration=%s",
        model.n_estimators,
        getattr(model, "best_iteration", None),
    )
    return model


def feature_importances(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted by gain (descending).

    Columns: feature, gain, gain_norm (sums to 1), weight (split count), cover.
    Features unused by any tree have gain=0 and sort to the bottom.
    """
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")

    rows = [
        {
            "feature": feat,
            "gain": gain.get(feat, 0.0),
            "weight": weight.get(feat, 0.0),
            "cover": cover.get(feat, 0.0),
        }
        for feat in feature_names
    ]
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    total = df["gain"].sum()
    df["gain_norm"] = (df["gain"] / total).round(4) if total > 0 else 0.0
    return df


def evaluate(
    model: XGBClassifier,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Return accuracy, f1, roc_auc, avg_precision. ROC metrics are nan if one class only."""
    if len(x) == 0:
        return {}
    x_df = pd.DataFrame(x, columns=feature_names)
    y_pred = model.predict(x_df)
    y_prob = model.predict_proba(x_df)[:, 1]  # type: ignore[index]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(set(y.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
    return metrics


def save_model(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logging.info("Saved model → %s", path)


def load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(path))
    return model
