"""
Generate binary token-importance labels matching kelle-simulator's AERP definition.

A token is labeled important (1) if at least 50% of attention heads vote to recompute
it — i.e., its cumulative attention received exceeds the eviction threshold in the AERP
policy. Labels are derived from attention traces and cached as .parquet files alongside
the feature table for fast reloading.
"""

# TODO: implement in follow-up session
