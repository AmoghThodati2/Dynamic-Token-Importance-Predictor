"""
scripts/01_collect_traces.py

Collect per-sample attention traces from SmolLM-135M for token-importance labeling.

Design decisions:
  eager attention:   SDPA and flash-attn do not return attention weights; eager does.
                     Passing attn_implementation="eager" is required — omitting it gives
                     out.attentions = None silently on most recent GPU-capable builds.
  per-sample files:  makes the run resumable and avoids loading a monolithic tensor into
                     memory when the downstream feature pipeline processes one sample at a
                     time.
  fp16 storage:      halves disk vs fp32 (~0.5 GB / 1000 samples at seq_len=256, 30
                     layers, 9 heads). Attention weights are probabilities in [0,1] so
                     fp16 precision is more than sufficient.
  JSONL metadata:    one append per sample, flushed immediately, so a crash never
                     corrupts completed entries.
  streaming dataset: avoids downloading the full corpus before starting; WikiText-103 is
                     ~500 MB uncompressed.
  C4 fallback:       C4 via HuggingFace requires gcsfs/s3fs for GCS streaming. If those
                     are absent, the script falls back to WikiText-103 automatically and
                     logs a warning naming the missing dependency.

Output (per sample):
  data/traces/sample_XXXXX.pt   — dict with keys: input_ids, attentions, source
  data/traces/metadata.jsonl    — one JSON line per sample (sample_id, seq_len, …)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
MIN_TEXT_CHARS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect SmolLM-135M attention traces over text samples."
    )
    p.add_argument("--num-samples", type=int, default=1000, metavar="N")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Truncation length in tokens. 512 is fine but uses ~4x the disk.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/traces"))
    p.add_argument(
        "--dataset",
        choices=["wikitext", "c4"],
        default="wikitext",
        help="Source corpus. C4 requires gcsfs; falls back to wikitext if unavailable.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level),
        stream=sys.stdout,
        force=True,
    )


def load_text_stream(dataset_name: str) -> tuple[Iterator[str], str]:
    """Return (text iterator, resolved_dataset_name).

    Falls back to wikitext automatically if C4 fails to load, logging a warning that
    names the missing dependency so the user knows what to install.
    """
    if dataset_name == "c4":
        try:
            ds = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            return (str(sample["text"]) for sample in ds), "c4"
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Could not load C4 (likely missing gcsfs or s3fs: %s). "
                "Install with: pip install gcsfs s3fs  — falling back to wikitext.",
                exc,
            )
            dataset_name = "wikitext"

    ds = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    return (str(sample["text"]) for sample in ds), "wikitext"


def load_model_and_tokenizer(
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logging.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logging.info("Loading model with attn_implementation='eager'")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    logging.info(
        "Model ready on %s — %d layers",
        device,
        model.config.num_hidden_layers,
    )
    return model, tokenizer


def existing_sample_ids(output_dir: Path) -> set[int]:
    return {int(p.stem.split("_")[1]) for p in output_dir.glob("sample_*.pt")}


def forward_pass(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: str,
) -> torch.Tensor | None:
    """Return stacked attention tensor (num_layers, num_heads, seq, seq) in fp16.

    Returns None if attentions are missing (e.g., wrong attn_implementation).
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            output_attentions=True,
            use_cache=False,
        )
    if out.attentions is None:
        return None
    # out.attentions: tuple of (batch, num_heads, seq, seq), one tensor per layer
    stacked = torch.stack(out.attentions, dim=0)  # (layers, batch, heads, seq, seq)
    stacked = stacked.squeeze(1)  # (layers, heads, seq, seq)
    return stacked.to(torch.float16).cpu()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_iter, resolved_dataset = load_text_stream(args.dataset)
    logging.info("Dataset: %s", resolved_dataset)

    model, tokenizer = load_model_and_tokenizer(args.device)

    done = existing_sample_ids(args.output_dir)
    logging.info(
        "Resuming: %d existing traces found; target is %d total.",
        len(done),
        args.num_samples,
    )

    metadata_path = args.output_dir / "metadata.jsonl"
    meta_file = metadata_path.open("a")

    sample_id = 0
    collected = len(done)
    pbar = tqdm(total=args.num_samples, initial=collected, desc="traces", unit="sample")

    try:
        for raw_text in text_iter:
            if collected >= args.num_samples:
                break

            if not raw_text or len(raw_text.strip()) < MIN_TEXT_CHARS:
                continue

            if sample_id in done:
                sample_id += 1
                continue

            try:
                enc = tokenizer(
                    raw_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_len,
                )
                input_ids: torch.Tensor = enc["input_ids"]
                seq_len = input_ids.shape[1]

                attn = forward_pass(model, input_ids, args.device)
                if attn is None:
                    logging.warning(
                        "Sample %05d: attentions is None — is attn_implementation set "
                        "to 'eager'? Skipping.",
                        sample_id,
                    )
                    sample_id += 1
                    continue

                out_path = args.output_dir / f"sample_{sample_id:05d}.pt"
                torch.save(
                    {
                        "input_ids": input_ids.cpu(),
                        "attentions": attn,
                        "source": resolved_dataset,
                    },
                    out_path,
                )

                meta: dict[str, object] = {
                    "sample_id": sample_id,
                    "source": resolved_dataset,
                    "dataset": resolved_dataset,
                    "seq_len": seq_len,
                    "num_layers": attn.shape[0],
                    "num_heads": attn.shape[1],
                    "model_name": MODEL_NAME,
                }
                meta_file.write(json.dumps(meta) + "\n")
                meta_file.flush()

                collected += 1
                pbar.update(1)
                logging.debug("Saved sample_%05d (seq_len=%d)", sample_id, seq_len)

            except Exception as exc:  # noqa: BLE001
                logging.warning("Sample %05d failed: %s — skipping.", sample_id, exc)

            sample_id += 1

    finally:
        pbar.close()
        meta_file.close()

    logging.info(
        "Done. Collected %d / %d traces in %s.",
        collected,
        args.num_samples,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
