"""
Extract per-token features from raw attention traces for the importance predictor.

Features include: position, recency, cumulative attention received, rolling-window
attention, attention trend, head-agreement fraction and variance, sink and recompute
classification flags, and cache-pressure metrics. All features are computed token-wise
over the stacked attention tensors produced by trace.py.
"""

# TODO: implement in follow-up session
