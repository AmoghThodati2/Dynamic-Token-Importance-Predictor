"""
Generate per-(layer, head, token) labels matching the simulator's per-head increment
condition in `systolic_evictor.accumulate_scores`.

For each (layer, head, token_pos) the label is:

    label = 1 if attn[l, h, T-1, k] > mean(attn[l, h, T-1, :]) else 0

This is the per-head decision underlying the AERP retention rule
(`head_counts / num_heads_total >= 0.5`): each head increments `head_counts[k]` whenever
the weight to token k exceeds the mean weight across all keys for that head's call. By
training the predictor against this exact per-head signal, the model learns to produce
distributions whose above-mean set matches the heuristic's above-mean set.

Output: one row per (sample_id, token_pos, layer, head) with columns:
    sample_id, token_pos, layer, head, label

Joinable on those four keys to features.parquet (per-head schema). The label table
always has the same row count as the feature table for the same trace set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from token_importance.trace import iter_traces


def compute_labels(attn: torch.Tensor, sample_id: int) -> pd.DataFrame:
    """
    Per-(layer, head, token_pos) above-mean labels for one sample.

    attn: (L, H, T, T) fp16 — attn[l, h, q, k] is weight query q gives to key k.
    Returns DataFrame with T*L*H rows ordered by (layer, head, token_pos), matching
    extract_features() row order so the two tables align without re-sorting.
    """
    n_layers, n_heads, seq_len, _ = attn.shape
    a = attn.float()

    last_attn_lh = a[:, :, -1, :]                            # (L, H, T)
    last_mean = last_attn_lh.mean(dim=2, keepdim=True)       # (L, H, 1)
    label_lhk = (last_attn_lh > last_mean).to(torch.int8)    # (L, H, T)

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
            "label": label_lhk.reshape(-1).numpy(),
        }
    )


def build_label_table(
    trace_dir: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Generate per-(layer, head, token_pos) above-mean labels for all traces in
    trace_dir and write a Parquet file at output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for i, (sid, _input_ids, attn, _meta) in enumerate(iter_traces(trace_dir)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            df = compute_labels(attn, sid)
            frames.append(df)
            logging.debug(
                "sample %05d: %d rows, %.1f%% positive",
                sid,
                len(df),
                100.0 * df["label"].mean(),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "sample %05d: label generation failed: %s — skipping", sid, exc
            )

    if not frames:
        msg = f"No labels generated from {trace_dir}"
        raise RuntimeError(msg)

    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(output_path, index=False)
    logging.info(
        "Label table: %d rows, %.1f%% positive → %s",
        len(table),
        100.0 * table["label"].mean(),
        output_path,
    )
    return table
