"""
Generate binary token-importance labels matching kelle-simulator's AERP definition.

For each (layer, head) pair, tokens are ranked by cumulative attention received
(sum of attention column over all query positions). A head "votes to retain" token k
if k is among the top-cache_size tokens by that metric. A token is labeled 1 (important)
if ≥vote_threshold fraction of all (layer, head) pairs vote to retain it — mirroring the
≥50% head-vote rule in kelle-simulator's AERP eviction policy.

Default cache_size = seq_len // 2, which yields approximately balanced classes and
matches the intuition that roughly half of a sequence's tokens are worth retaining.
At the reference simulator cache size of N=128, use --cache-size 128 when seq_len > 128.

Output columns:
    sample_id   — from the source trace
    token_pos   — key position in [0, seq_len)
    label       — 1 if retained by ≥50% of heads, else 0
    n_votes     — raw vote count (number of (layer, head) pairs that retained this token)
    vote_frac   — n_votes / (num_layers × num_heads)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from token_importance.trace import iter_traces


def compute_labels(
    attn: torch.Tensor,
    sample_id: int,
    cache_size: int | None = None,
    vote_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute per-token AERP labels for one sample.

    attn:           (L, H, T, T) fp16 — attn[l, h, q, k] is weight query q gives to key k.
    cache_size:     tokens each head retains (default: T // 2). Clamped to [1, T].
    vote_threshold: minimum fraction of heads that must vote to retain for label=1.
    """
    n_layers, n_heads, seq_len, _ = attn.shape
    a = attn.float()

    k = cache_size if cache_size is not None else max(seq_len // 2, 1)
    k = max(1, min(k, seq_len))

    if k == seq_len:
        logging.warning(
            "sample %05d: cache_size (%d) >= seq_len (%d) — all tokens labeled positive",
            sample_id,
            k,
            seq_len,
        )

    # Cumulative attention received by each token per (layer, head)
    col_sums = a.sum(dim=2)  # (L, H, T): sum over query dim

    # For each (layer, head), identify the top-k tokens by cumulative attention
    _, top_idx = col_sums.topk(k, dim=2)  # (L, H, k)

    # Scatter-count: how many (layer, head) pairs retained each token
    votes = torch.zeros(seq_len, dtype=torch.float32)
    flat_idx = top_idx.reshape(-1)  # (L * H * k,)
    votes.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

    n_total = n_layers * n_heads
    vote_frac = votes / n_total
    labels = (vote_frac >= vote_threshold).to(torch.int8)

    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "token_pos": torch.arange(seq_len).numpy(),
            "label": labels.numpy(),
            "n_votes": votes.to(torch.int32).numpy(),
            "vote_frac": vote_frac.numpy(),
        }
    )


def build_label_table(
    trace_dir: Path,
    output_path: Path,
    cache_size: int | None = None,
    vote_threshold: float = 0.5,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Generate labels for all traces in trace_dir and write a Parquet file at output_path.

    Processes one trace at a time. The output can be joined to the feature table from
    features.py on (sample_id, token_pos) for downstream XGBoost / MLP training.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for i, (sid, _input_ids, attn, _meta) in enumerate(iter_traces(trace_dir)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            df = compute_labels(
                attn, sid, cache_size=cache_size, vote_threshold=vote_threshold
            )
            frames.append(df)
            logging.debug(
                "sample %05d: %d tokens, %.1f%% positive",
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

    pos_rate = table["label"].mean()
    logging.info(
        "Label table: %d rows, %.1f%% positive → %s",
        len(table),
        100.0 * pos_rate,
        output_path,
    )
    return table
