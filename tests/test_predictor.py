"""End-to-end smoke tests for TokenImportancePredictor.

Loads the real artifacts in data/models/ and verifies the contract that
kelle-simulator's `_head_attention_weights` slot expects:

  - keys(output) == set(sorted_tids)
  - sum(values) == 1.0 (within fp tolerance) for non-empty input
  - 0 <= v <= 1 for all v
  - empty input → {}
  - same sorted_tids, different (layer, head) → different output
    (load-bearing for AERP head-vote diversity)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from token_importance.predictor import TokenImportancePredictor

MODELS_DIR = Path("data/models")
ARTIFACTS_PRESENT = all(
    (MODELS_DIR / f).exists() for f in ("mlp.pt", "scaler.pkl", "feature_schema.json")
)
pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="trained model artifacts not present in data/models/",
)


@pytest.fixture(scope="module")
def predictor() -> TokenImportancePredictor:
    # Stage-2 target shape (OPT-125M) — exercises the n_layers/n_heads
    # constructor inputs that differ from training (SmolLM-135M, 30 × 9).
    return TokenImportancePredictor(n_layers=12, n_heads=12)


def _sample_tids(seq_len: int = 128, after_eviction: bool = False) -> list[int]:
    if not after_eviction:
        return list(range(seq_len))
    # Drop a swath of mid-sequence tokens to mimic post-eviction state.
    return [i for i in range(seq_len) if i < 4 or i >= seq_len - 16 or i % 5 == 0]


def test_keys_match_input(predictor: TokenImportancePredictor) -> None:
    tids = _sample_tids(128)
    out = predictor(tids, layer=3, head=7, initial_preserved=10, recent_window=64)
    assert set(out.keys()) == set(tids)


def test_weights_sum_to_one(predictor: TokenImportancePredictor) -> None:
    tids = _sample_tids(128)
    out = predictor(tids, layer=3, head=7, initial_preserved=10, recent_window=64)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-6)


def test_weights_in_unit_interval(predictor: TokenImportancePredictor) -> None:
    tids = _sample_tids(128)
    out = predictor(tids, layer=3, head=7, initial_preserved=10, recent_window=64)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_empty_input_returns_empty_dict(predictor: TokenImportancePredictor) -> None:
    assert predictor([], layer=0, head=0, initial_preserved=4, recent_window=64) == {}


def test_layer_head_changes_output(predictor: TokenImportancePredictor) -> None:
    tids = _sample_tids(128)
    a = predictor(tids, layer=0, head=0, initial_preserved=10, recent_window=64)
    b = predictor(tids, layer=5, head=5, initial_preserved=10, recent_window=64)
    # Pointwise difference on at least one shared key — head-vote diversity is
    # the load-bearing AERP property; if (layer, head) is ignored the simulator's
    # head_counts / num_heads_total >= 0.5 rule collapses.
    differing = [k for k in tids if abs(a[k] - b[k]) > 1e-7]
    assert differing, "predictor output identical across (layer, head) — head-vote diversity lost"


def test_post_eviction_input(predictor: TokenImportancePredictor) -> None:
    tids = _sample_tids(256, after_eviction=True)
    out = predictor(tids, layer=2, head=4, initial_preserved=10, recent_window=64)
    assert set(out.keys()) == set(tids)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-6)


def test_singleton_input(predictor: TokenImportancePredictor) -> None:
    out = predictor([0], layer=0, head=0, initial_preserved=10, recent_window=64)
    assert out == {0: 1.0}


def test_call_alias() -> None:
    # `predictor._head_attention_weights(...)` matches the simulator's function
    # name and must produce the same output as `predictor(...)`.
    p = TokenImportancePredictor(n_layers=12, n_heads=12)
    tids = list(range(64))
    a = p(tids, 1, 2, 10, 64)
    b = p._head_attention_weights(tids, 1, 2, 10, 64)
    assert a == b
