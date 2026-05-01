"""
Load and preprocess raw attention trace files written by scripts/01_collect_traces.py.

Provides utilities to iterate over sample_XXXXX.pt files, verify their shapes, and
yield (input_ids, attentions, metadata) tuples ready for feature extraction. Also
handles cache-compatible reshaping and validation of the stored fp16 attention tensors.
"""

# TODO: implement in follow-up session
