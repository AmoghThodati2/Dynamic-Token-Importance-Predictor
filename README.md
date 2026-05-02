# Dynamic Token Importance Predictor

A learned per-(layer, head) attention-weight predictor that drops into
[kelle-simulator](https://github.com/LonghornSilicon/kelle-simulator)'s
`_head_attention_weights` slot. Collects attention traces from SmolLM-135M, extracts
position-only features per `(sample_id, token_pos, layer, head)`, labels each
(layer, head) above-mean retention vote, and trains a small MLP that outputs
normalised `{token_id: weight}` distributions consumed by AERP eviction.

## Status

| Stage | Description | State |
|-------|-------------|-------|
| **Stage 1** | Standalone PyTorch module (this repo): trace collection, feature engineering, XGBoost ranking, MLP training, simulator-shaped predictor | **Complete** |
| **Stage 2** | Public fork of [kelle-simulator](https://github.com/LonghornSilicon/kelle-simulator) importing this module; `--sweep-kv-capacity` to find smallest N' matching heuristic baseline at N'=128 | Planned |

## Pipeline

| # | Script | Output |
|---|--------|--------|
| 1 | `scripts/01_collect_traces.py` | `data/traces/sample_*.pt` — 1,000 SmolLM-135M attention tensors `(30, 9, T, T)` |
| 2 | `scripts/02_extract_features.py` | `data/features/features.parquet` — 44,470,890 rows × 11 cols, key `(sample_id, token_pos, layer, head)` |
| 3 | `scripts/03_generate_labels.py` | `data/features/labels.parquet` — per-head above-mean labels, 10.4% positive |
| 4 | `scripts/04_train_xgb.py` | `data/models/xgb_selector.json` — feature ranker, **test AUC 0.8008** |
| 5 | `scripts/05_train_mlp.py` | `data/models/{mlp.pt, scaler.pkl, feature_schema.json}` — **test AUC 0.7830** |

Trace and model artifacts are gitignored. Splits are by `sample_id` (seed=42, val/test=0.15).

## Features (single source: `src/token_importance/features.py`)

`pos`, `rel_pos`, `recency`, `is_sink`, `is_recent`, `rel_layer`, `rel_head`

All seven are inference-realistic: the simulator only provides `(sorted_tids,
layer, head, initial_preserved, recent_window)`, so per-head attention statistics
are intentionally excluded — using them would leak the above-mean label.

XGBoost gain ranking: recency 0.354, pos 0.232, rel_pos 0.222, rel_layer 0.130,
rel_head 0.046, is_recent 0.015, is_sink 0.000.

## Predictor

`src/token_importance/predictor.py:TokenImportancePredictor`. Signature is
identical to kelle-simulator's `_head_attention_weights`:

```python
predictor = TokenImportancePredictor(n_layers=12, n_heads=12)
weights = predictor(sorted_token_ids, layer, head, initial_preserved, recent_window)
# {token_id: float} with sum == 1.0
```

Constructor loads `data/models/{mlp.pt, scaler.pkl, feature_schema.json}` once.
Inference is ~82 µs/call on CPU at 128 cached tokens — within the simulator's
per-step budget at 12L × 12H = 144 head-calls per decode step.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

`tests/test_predictor.py` covers the simulator contract: keys match input,
weights sum to 1, range [0, 1], empty/singleton input, post-eviction sparse
input, and `(layer, head)` output diversity (load-bearing for AERP head votes).

## Known limitations / distribution shifts at Stage 2

These are flagged in code; none are silently corrected.

- **Layout shift**: training was SmolLM-135M (30 layers × 9 heads); Stage 2 target
  is OPT-125M (12 × 12). `rel_layer` and `rel_head` stay in [0, 1] but their joint
  distribution differs.
- **Recent-window shift**: training used `recent_window=8`; simulator default is
  64. The `is_recent` flag fires on many more positions at inference.
- **Position extrapolation**: training `pos` was bounded by trace length (≤ 256);
  long generations push `pos` beyond this, and the StandardScaler extrapolates
  linearly.
- **CPU-only inference**: the MLP is sized for CPU deployment inside the
  simulator's decode loop; no GPU dependency at inference time.

## License

MIT
