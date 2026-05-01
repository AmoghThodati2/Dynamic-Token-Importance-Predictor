"""
XGBoost wrapper for feature importance ranking and baseline classification.

Trains an XGBClassifier on the full feature table, extracts feature importances to
identify the top-K features for MLP input, and serializes the fitted model to
data/models/xgb_selector.json. Also used as a standalone baseline to compare against
the MLP predictor.
"""

# TODO: implement in follow-up session
