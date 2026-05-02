"""
TokenImportancePredictor — drop-in replacement for kelle-simulator's
`_head_attention_weights(sorted_tids, layer, head, initial_preserved, recent_window)`.

Loads the trained MLP (data/models/mlp.pt) plus its StandardScaler
(data/models/scaler.pkl) and feature schema (data/models/feature_schema.json) once at
construction, then on each call builds a per-token feature matrix matching
features.py exactly, runs forward + sigmoid + L1 normalization, and returns a
{token_id: weight} dict consumed by SystolicEvictor.accumulate_scores.

Distribution-shift caveats vs. training (SmolLM-135M, T <= 256, recent_window=8):
  - Stage 2 target OPT-125M is 12 layers × 12 heads (training was 30 × 9). The
    rel_layer / rel_head inputs occupy the same [0, 1] range, but their joint
    distribution differs.
  - Simulator default recent_window is 64; training used 8. The is_recent flag
    fires for many more positions at inference.
  - Absolute positions can exceed the training max (256) once a long generation
    runs; the StandardScaler extrapolates linearly, so the model is operating
    out-of-distribution on `pos` past that point.
None of these are silently corrected — they are the cost of porting a
position-only predictor to a longer-context simulator and are flagged so any
Stage 2 quality regression is attributable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from token_importance.features import FEATURE_COLS
from token_importance.models.mlp import ImportanceModel

DEFAULT_MODELS_DIR = Path("data/models")


class TokenImportancePredictor:
    """Per-(layer, head) attention-weight predictor with the simulator signature."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
    ) -> None:
        models_dir = Path(models_dir)

        schema = json.loads((models_dir / "feature_schema.json").read_text())
        if schema["feature_cols"] != FEATURE_COLS:
            msg = (
                f"feature_schema.json columns {schema['feature_cols']} do not match "
                f"features.FEATURE_COLS {FEATURE_COLS}"
            )
            raise RuntimeError(msg)

        with (models_dir / "scaler.pkl").open("rb") as f:
            self._scaler = pickle.load(f)
        if self._scaler.n_features_in_ != len(FEATURE_COLS):
            msg = (
                f"scaler expects {self._scaler.n_features_in_} features but "
                f"FEATURE_COLS has {len(FEATURE_COLS)}"
            )
            raise RuntimeError(msg)

        model = ImportanceModel(input_dim=len(FEATURE_COLS))
        state = torch.load(
            models_dir / "mlp.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        self._model = model

        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)

        # Pre-compute scaler params as float32 tensors for cheap inference math.
        self._scaler_mean = torch.from_numpy(
            self._scaler.mean_.astype(np.float32)
        )
        self._scaler_scale = torch.from_numpy(
            self._scaler.scale_.astype(np.float32)
        )

    @torch.no_grad()
    def __call__(
        self,
        sorted_token_ids: list[int],
        layer: int,
        head: int,
        initial_preserved: int,  # noqa: ARG002 (signature parity with simulator)
        recent_window: int,
    ) -> dict[int, float]:
        if not sorted_token_ids:
            return {}

        tids = torch.as_tensor(sorted_token_ids, dtype=torch.float32)

        # `pos` at inference is the absolute sequence position carried by
        # aerp._tokens.keys() — verified via simulator.py:321-322,343-345 where
        # tok_id IS the position. seq_len = max(pos) + 1 keeps rel_pos / recency
        # on the same absolute-position scale as training (where the trace had
        # no eviction so max(pos)+1 == len(positions)). Using len(sorted_tids)
        # would inflate rel_pos once eviction creates gaps.
        pos = tids
        seq_len = float(tids.max().item()) + 1.0
        rel_pos = pos / max(seq_len - 1.0, 1.0)
        recency = 1.0 - rel_pos
        is_sink = (pos == 0.0).to(torch.float32)
        pos_from_end = (seq_len - 1.0) - pos
        is_recent = (pos_from_end < float(recent_window)).to(torch.float32)

        n = pos.shape[0]
        rel_layer = torch.full(
            (n,), layer / max(self.n_layers - 1, 1), dtype=torch.float32
        )
        rel_head = torch.full(
            (n,), head / max(self.n_heads - 1, 1), dtype=torch.float32
        )

        x = torch.stack(
            [pos, rel_pos, recency, is_sink, is_recent, rel_layer, rel_head],
            dim=1,
        )
        x = (x - self._scaler_mean) / self._scaler_scale

        logits = self._model(x)
        probs = torch.sigmoid(logits)
        total = probs.sum()
        # Fall back to uniform if all sigmoids underflowed; accumulate_scores
        # still needs a normalised distribution.
        weights = torch.full_like(probs, 1.0 / n) if total <= 0 else probs / total

        weights_list = weights.tolist()
        return {
            int(tid): float(w)
            for tid, w in zip(sorted_token_ids, weights_list, strict=True)
        }

    # Convenience alias matching the function name in kelle-simulator.
    def _head_attention_weights(
        self,
        sorted_token_ids: list[int],
        layer: int,
        head: int,
        initial_preserved: int,
        recent_window: int,
    ) -> dict[int, float]:
        return self(sorted_token_ids, layer, head, initial_preserved, recent_window)
