"""Loads the trained model and scores customers."""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features import REASON_GROUPS, make_features

FOLDER = Path(__file__).resolve().parent  # all files sit in one folder

# ---- Loaded once, when the API starts ----
model = lgb.Booster(model_file=str(FOLDER / "model.txt"))
model_info = json.loads((FOLDER / "model_info.json").read_text())
FEATURES = model.feature_name()
THRESHOLD = model_info["threshold"]


def predict(customer: dict) -> dict:
    """Score one customer. `customer` has the same keys as make_features() expects."""
    X = make_features(pd.DataFrame([customer]))[FEATURES]

    probability = float(model.predict(X)[0])

    # pred_contrib=True asks LightGBM how much each feature pushed the score
    # up or down for THIS customer (these are SHAP values). The last column
    # is the starting average, so we drop it.
    contributions = model.predict(X, pred_contrib=True)[0][:-1]
    by_feature = dict(zip(FEATURES, contributions))

    # Add the contributions up within each reason group.
    reasons = []
    for reason, columns in REASON_GROUPS.items():
        total = sum(by_feature[c] for c in columns)
        reasons.append({"reason": reason, "effect": "raises risk" if total > 0 else "lowers risk", "strength": round(abs(total), 3)})

    # Keep the 3 strongest reasons.
    reasons.sort(key=lambda r: r["strength"], reverse=True)

    return {
        "probability_of_default": round(probability, 3),
        "decision": "decline" if probability >= THRESHOLD else "approve",
        "top_reasons": reasons[:3],
    }
