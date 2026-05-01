"""
Command-line entry points for the token-importance pipeline.

Exposes three commands registered in pyproject.toml:
  tip-collect  — wraps scripts/01_collect_traces.py for package-level invocation
  tip-train    — runs XGBoost feature selection then MLP training end-to-end
  tip-eval     — evaluates the trained predictor against the kelle-simulator baseline
"""

# TODO: implement in follow-up session
