"""
Reusable PyTorch module wrapping the trained MLP token-importance predictor.

Exposes a clean interface: given a (num_layers, num_heads, seq_len, seq_len) attention
tensor and token position, returns a (seq_len,) tensor of importance scores in [0, 1].
Designed for CPU-only inference and direct integration into the kelle-simulator decode
loop as a drop-in replacement for the uniform attention-weight placeholder.
"""

# TODO: implement in follow-up session
