"""
Extract per-(layer, head, token) features from raw attention traces.

For each sample with attention shape (L, H, T, T), emits T*L*H rows uniquely keyed by
(sample_id, token_pos, layer, head) — matching the call pattern of kelle-simulator's
`_head_attention_weights(sorted_tids, layer, head, ...)`.

All features are inference-realistic — the predictor at simulator integration time
only receives `(sorted_tids, layer, head, initial_preserved, recent_window)` and has
no access to attention tensors. Per-head attention statistics are intentionally
excluded: under the labels.py above-mean rule, `last_attn_lh > mean(last_attn_lh)`
IS the label, and including any attention statistic as a feature would leak.

Feature breakdown:

- Token-level (broadcast across layer × head):
    pos, rel_pos, recency, is_sink, is_recent (recent_window=8 default)

- Head-context (varying per row):
    rel_layer = layer / max(n_layers - 1, 1)
    rel_head  = head  / max(n_heads  - 1, 1)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from token_importance.trace import iter_traces

RECENT_WINDOW = 8

FEATURE_COLS = [
    "pos",
    "rel_pos",
    "recency",
    "is_sink",
    "is_recent",
    "rel_layer",
    "rel_head",
]


def extract_features(
    attn: torch.Tensor,
    sample_id: int,
    recent_window: int = RECENT_WINDOW,
) -> pd.DataFrame:
    """
    Per-(layer, head, token_pos) features for one sample.

    attn:          (L, H, T, T) — only the shape is consulted; values unused.
    sample_id:     propagated to the 'sample_id' column.
    recent_window: 1 if pos_from_end < recent_window else 0 (matches simulator semantic).

    Returns DataFrame with T*L*H rows, ordered by (layer, head, token_pos) — i.e.,
    row r corresponds to (l = r // (H*T), h = (r // T) % H, k = r % T).
    """
    n_layers, n_heads, seq_len, _ = attn.shape

    pos = torch.arange(seq_len, dtype=torch.float32)
    rel_pos = pos / float(max(seq_len - 1, 1))
    recency = 1.0 - rel_pos
    is_sink = (pos == 0.0).float()
    pos_from_end = float(seq_len - 1) - pos
    is_recent = (pos_from_end < float(recent_window)).float()

    rel_layer = torch.arange(n_layers, dtype=torch.float32) / float(max(n_layers - 1, 1))
    rel_head = torch.arange(n_heads, dtype=torch.float32) / float(max(n_heads - 1, 1))

    lh = n_layers * n_heads
    layer_arr = torch.arange(n_layers, dtype=torch.int32).repeat_interleave(n_heads * seq_len)
    head_arr = torch.arange(n_heads, dtype=torch.int32).repeat_interleave(seq_len).repeat(n_layers)
    token_pos_arr = torch.arange(seq_len, dtype=torch.int32).repeat(lh)

    return pd.DataFrame(
        {
            "sample_id": np.full(lh * seq_len, sample_id, dtype=np.int32),
            "token_pos": token_pos_arr.numpy(),
            "layer": layer_arr.numpy(),
            "head": head_arr.numpy(),
            "pos": pos.repeat(lh).numpy(),
            "rel_pos": rel_pos.repeat(lh).numpy(),
            "recency": recency.repeat(lh).numpy(),
            "is_sink": is_sink.repeat(lh).numpy(),
            "is_recent": is_recent.repeat(lh).numpy(),
            "rel_layer": rel_layer.repeat_interleave(n_heads * seq_len).numpy(),
            "rel_head": rel_head.repeat_interleave(seq_len).repeat(n_layers).numpy(),
        }
    )


def build_feature_table(
    trace_dir: Path,
    output_path: Path,
    recent_window: int = RECENT_WINDOW,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Extract per-(layer, head, token) features from all traces in trace_dir and write
    a Parquet file at output_path.

    Output is one row per (sample_id, token_pos, layer, head). Joinable to per-head
    labels (labels.py) on those four keys.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for i, (sid, _input_ids, attn, _meta) in enumerate(iter_traces(trace_dir)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            df = extract_features(attn, sid, recent_window=recent_window)
            frames.append(df)
            logging.debug("sample %05d: extracted %d (l,h,k) rows", sid, len(df))
        except Exception as exc:  # noqa: BLE001
            logging.warning("sample %05d: feature extraction failed: %s — skipping", sid, exc)

    if not frames:
        msg = f"No features extracted from {trace_dir}"
        raise RuntimeError(msg)

    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(output_path, index=False)
    logging.info(
        "Feature table: %d rows × %d cols → %s",
        len(table),
        len(table.columns),
        output_path,
    )
    return table
