"""
Load and preprocess raw attention trace files written by scripts/01_collect_traces.py.

Provides utilities to iterate over sample_XXXXX.pt files, verify their shapes, and
yield (sample_id, input_ids, attentions, metadata) tuples ready for feature extraction.
Skips and logs any file that fails to load or has an unexpected attention shape.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import torch


def load_trace(path: Path) -> dict:
    return torch.load(path, weights_only=True)


def load_metadata(trace_dir: Path) -> dict[int, dict]:
    """Return metadata.jsonl contents keyed by sample_id. Empty dict if file missing."""
    meta_path = trace_dir / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    meta: dict[int, dict] = {}
    for line in meta_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            meta[int(row["sample_id"])] = row
    return meta


def validate_attentions(attn: torch.Tensor, sample_id: int) -> bool:
    if attn.ndim != 4:
        logging.warning(
            "sample %05d: attentions has %d dims, expected 4 — skipping", sample_id, attn.ndim
        )
        return False
    _n_layers, _n_heads, t1, t2 = attn.shape
    if t1 != t2:
        logging.warning(
            "sample %05d: attention matrix not square (%d × %d) — skipping", sample_id, t1, t2
        )
        return False
    return True


def iter_traces(
    trace_dir: Path,
    sample_ids: set[int] | None = None,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, dict]]:
    """
    Yield (sample_id, input_ids, attentions, metadata) for each valid trace in trace_dir.

    attentions: (num_layers, num_heads, seq_len, seq_len) fp16
    input_ids:  (1, seq_len) int64
    """
    metadata = load_metadata(trace_dir)
    for path in sorted(trace_dir.glob("sample_*.pt")):
        sid = int(path.stem.split("_")[1])
        if sample_ids is not None and sid not in sample_ids:
            continue
        try:
            data = load_trace(path)
        except Exception:  # noqa: BLE001
            logging.warning("Failed to load %s — skipping", path)
            continue
        attn = data["attentions"]
        if not validate_attentions(attn, sid):
            continue
        yield sid, data["input_ids"], attn, metadata.get(sid, {})
