"""
Small MLP token-importance predictor: linear(input_dim → 32) → ReLU → linear(32 → 16)
→ ReLU → linear(16 → 1) → sigmoid.

Input dimension is determined by the XGBoost feature selection step. Trained with
BCELoss and AdamW, and serialized to data/models/mlp.pt for use in predictor.py.
"""

# TODO: implement in follow-up session
