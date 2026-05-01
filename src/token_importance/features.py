"""
Extract per-token features from raw attention traces for the importance predictor.

For each token at key position k in a sequence of length T, features are derived from
the column attn[:, :, :, k] of the (L, H, T, T) attention tensor — i.e., how much
attention every query gave to this token, broken down by layer and head. Features
cover: position/recency, cumulative attention, rolling-window attention, cross-layer
trend, and head-agreement statistics. All computed in fp32 from fp16 stored tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from token_importance.trace import iter_traces

WINDOW_SIZE = 8

FEATURE_COLS = [
    "pos",             # absolute token position (float)
    "rel_pos",         # pos / (T-1), normalized 0..1
    "recency",         # 1 - rel_pos; 0 = most recent token, 1 = oldest
    "is_sink",         # 1.0 if pos == 0 (attention-sink indicator), else 0.0
    "cum_attn",        # mean over (L,H) of Σ_q attn[l,h,q,k] — cumulative attention received
    "last_attn",       # mean over (L,H) of attn[l,h,T-1,k] — attention from final query
    "window_attn",     # mean over (L,H) and last WINDOW_SIZE queries of attn[l,h,q,k]
    "attn_trend",      # OLS slope of per-layer mean last-query attention across layer depth
    "head_agree_frac", # fraction of (L,H) pairs where last-query attn > uniform baseline 1/T
    "head_agree_var",  # variance of last-query attention across all (L,H) pairs
    "max_layer_attn",  # max over layers of mean-head last-query attention
    "min_layer_attn",  # min over layers of mean-head last-query attention
]


def extract_features(
    attn: torch.Tensor,
    sample_id: int,
    window: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Compute per-token features for one sample.

    attn:      (L, H, T, T) fp16 — attn[l, h, q, k] is the weight query q gives to key k.
    sample_id: propagated to the 'sample_id' column.
    Returns a DataFrame of shape (T, 2 + len(FEATURE_COLS)) with sample_id and token_pos
    as identifier columns followed by all feature columns.
    """
    n_layers, n_heads, seq_len, _ = attn.shape
    a = attn.float()  # fp32 for numerical stability

    # ---- per-(layer, head, key-token) aggregations ----
    cum_attn_lhk = a.sum(dim=2)                      # (L, H, T)  sum over query dim
    last_attn_lhk = a[:, :, -1, :]                   # (L, H, T)  last-query row
    w = min(window, seq_len)
    win_attn_lhk = a[:, :, -w:, :].mean(dim=2)       # (L, H, T)  rolling window mean

    # ---- scalar features per key token (T,) ----
    cum_attn = cum_attn_lhk.mean(dim=(0, 1))
    last_attn = last_attn_lhk.mean(dim=(0, 1))
    window_attn = win_attn_lhk.mean(dim=(0, 1))

    # Trend: OLS slope of mean-head last-query attention vs layer index
    per_layer_last = last_attn_lhk.mean(dim=1)       # (L, T)
    layer_idx = torch.arange(n_layers, dtype=torch.float32)
    lc = layer_idx - layer_idx.mean()                # centred layer indices (L,)
    layer_var = (lc**2).sum()
    per_layer_centred = per_layer_last - per_layer_last.mean(dim=0, keepdim=True)  # (L, T)
    attn_trend = (lc[:, None] * per_layer_centred).sum(dim=0) / (layer_var + 1e-8)

    max_layer_attn = per_layer_last.max(dim=0).values
    min_layer_attn = per_layer_last.min(dim=0).values

    # Head agreement relative to the uniform-attention baseline 1/T
    threshold = 1.0 / max(seq_len, 1)
    flat_last = last_attn_lhk.reshape(n_layers * n_heads, seq_len)
    head_agree_frac = (flat_last > threshold).float().mean(dim=0)
    head_agree_var = flat_last.var(dim=0)

    # Position features
    pos = torch.arange(seq_len, dtype=torch.float32)
    denom = float(max(seq_len - 1, 1))
    rel_pos = pos / denom
    recency = 1.0 - rel_pos
    is_sink = (pos == 0.0).float()

    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "token_pos": torch.arange(seq_len).numpy(),
            "pos": pos.numpy(),
            "rel_pos": rel_pos.numpy(),
            "recency": recency.numpy(),
            "is_sink": is_sink.numpy(),
            "cum_attn": cum_attn.numpy(),
            "last_attn": last_attn.numpy(),
            "window_attn": window_attn.numpy(),
            "attn_trend": attn_trend.numpy(),
            "head_agree_frac": head_agree_frac.numpy(),
            "head_agree_var": head_agree_var.numpy(),
            "max_layer_attn": max_layer_attn.numpy(),
            "min_layer_attn": min_layer_attn.numpy(),
        }
    )


def build_feature_table(
    trace_dir: Path,
    output_path: Path,
    window: int = WINDOW_SIZE,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Extract features from all traces in trace_dir and write a Parquet file at output_path.

    Processes one trace at a time to keep peak memory proportional to a single
    (L, H, T, T) tensor rather than the full corpus.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for i, (sid, _input_ids, attn, _meta) in enumerate(iter_traces(trace_dir)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            df = extract_features(attn, sid, window=window)
            frames.append(df)
            logging.debug("sample %05d: extracted %d token rows", sid, len(df))
        except Exception as exc:  # noqa: BLE001
            logging.warning("sample %05d: feature extraction failed: %s — skipping", sid, exc)

    if not frames:
        msg = f"No features extracted from {trace_dir}"
        raise RuntimeError(msg)

    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(output_path, index=False)
    logging.info(
        "Feature table: %d rows × %d cols → %s", len(table), len(table.columns), output_path
    )
    return table
