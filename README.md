# Dynamic Token Importance Predictor

A learned token-importance predictor for KV cache recompute decisions. Collects attention
traces from SmolLM-135M, builds a feature table over per-token attention statistics, and
trains a small MLP to predict which tokens a KV cache eviction policy will need to
recompute — replacing the uniform-weight placeholder in the kelle-simulator decode loop.

## Status

| Stage | Description | State |
|-------|-------------|-------|
| **Stage 1** | Standalone PyTorch module (this repo): trace collection, feature engineering, XGBoost selection, MLP training | **In progress** |
| **Stage 2** | Public fork of [kelle-simulator](https://github.com/LonghornSilicon/kelle-simulator) importing this module; `--sweep-kv-capacity` to find smallest N' matching heuristic baseline at N'=128 | Planned |

## Pipeline

| # | Script | Status |
|---|--------|--------|
| 1 | `scripts/01_collect_traces.py` — forward pass SmolLM-135M, save `(layers, heads, seq, seq)` attention tensors | **Implemented** |
| 2 | `scripts/02_extract_features.py` — extract 12-feature per-token table → `data/features/features.parquet` | **Implemented** |
| 3 | `scripts/03_generate_labels.py` — AERP binary labels → `data/features/labels.parquet` | **Implemented** |
| 4 | `scripts/04_train_xgb.py` — XGBoost feature ranking → `data/models/xgb_selector.json` + `selected_features.json` | **Implemented** |
| 5 | `tip-train` (MLP) — train `input → 32 → 16 → 1 sigmoid` on pruned feature set | Planned |
| 6 | `tip-eval` — evaluate predictor against kelle-simulator AERP baseline | Planned |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Known limitations

- **Model mismatch**: traces are collected from SmolLM-135M; the kelle-simulator targets
  OPT-125M. Attention-based features should generalize across architectures but verify
  prediction quality in Stage 2 before relying on the predictor.
- **Label definition**: binary labels mirror kelle-simulator's AERP recompute rule —
  a token is "important" if ≥50% of attention heads vote to retain it at eviction time.
- **CPU-only inference target**: the MLP is designed for CPU deployment inside the
  simulator's decode loop; no GPU dependency at inference time.

## License

MIT
