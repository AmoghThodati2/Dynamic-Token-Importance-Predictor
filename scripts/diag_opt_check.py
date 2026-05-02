"""
scripts/diag_opt_check.py

Stage-2 sanity check: do features computed from OPT-125M attention look qualitatively
similar to features from SmolLM-135M? The simulator targets OPT-125M (12 layers × 12
full attention heads, no GQA), while training data is SmolLM-135M (30 layers × 9 query
heads with GQA → 3 KV heads). If the feature distributions diverge sharply, cross-model
transfer is broken before Stage 2 even starts.

What this script does:
  1. Loads facebook/opt-125m with attn_implementation='eager'
  2. Runs forward passes on a handful of WikiText samples
  3. Verifies attention shape == (12, 12, T, T)
  4. Computes features via token_importance.features.extract_features
  5. Prints describe() side-by-side with the SmolLM features for comparison

This does not save anything to disk. It's a one-shot diagnostic; re-run when you want
a fresh comparison.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OPT_MODEL = "facebook/opt-125m"
EXPECTED_LAYERS = 12
EXPECTED_HEADS = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPT-125M attention sanity check.")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument(
        "--smollm-features-path",
        type=Path,
        default=Path("data/features/features.parquet"),
        help="Existing SmolLM feature table to compare distributions against.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
        force=True,
    )

    from token_importance.features import FEATURE_COLS, extract_features

    logging.info("Loading %s with eager attention", OPT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(OPT_MODEL, attn_implementation="eager")
    model.eval()

    cfg = model.config
    print(f"\nOPT-125M config: layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}, "
          f"d_model={cfg.hidden_size}")
    if cfg.num_hidden_layers != EXPECTED_LAYERS or cfg.num_attention_heads != EXPECTED_HEADS:
        logging.warning(
            "Unexpected OPT config: got %d layers x %d heads (expected %d x %d)",
            cfg.num_hidden_layers, cfg.num_attention_heads, EXPECTED_LAYERS, EXPECTED_HEADS,
        )

    logging.info("Streaming wikitext for %d samples", args.num_samples)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    frames: list[pd.DataFrame] = []
    seen = 0
    for raw in ds:
        if seen >= args.num_samples:
            break
        text = (raw.get("text") or "").strip()
        if len(text) < 200:
            continue
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.max_seq_len
        )
        input_ids = enc["input_ids"]

        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)

        if out.attentions is None:
            logging.error("attentions is None — check attn_implementation='eager'")
            sys.exit(1)

        stacked = torch.stack(out.attentions, dim=0).squeeze(1).to(torch.float16).cpu()
        if seen == 0:
            print(f"OPT attention shape: {tuple(stacked.shape)} (per sample)")
        frames.append(extract_features(stacked, sample_id=seen))
        seen += 1

    if not frames:
        logging.error("No usable samples — bailing out")
        sys.exit(1)

    opt_feats = pd.concat(frames, ignore_index=True)

    print(f"\nOPT-125M ({seen} samples × seq_len={args.max_seq_len}) feature stats:")
    print(opt_feats[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].to_string())

    if args.smollm_features_path.exists():
        smollm_feats = pd.read_parquet(args.smollm_features_path)
        print(f"\nSmolLM-135M ({smollm_feats['sample_id'].nunique()} samples) "
              f"feature stats — for side-by-side comparison:")
        print(smollm_feats[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].to_string())

        print("\nMean ratio (OPT / SmolLM) per feature — values near 1.0 indicate good transfer:")
        opt_means = opt_feats[FEATURE_COLS].mean()
        smol_means = smollm_feats[FEATURE_COLS].mean()
        ratios = (opt_means / smol_means.replace(0, float("nan"))).round(3)
        print(ratios.to_string())
    else:
        print(f"\n(No SmolLM feature table at {args.smollm_features_path} — skipping comparison.)")


if __name__ == "__main__":
    main()
