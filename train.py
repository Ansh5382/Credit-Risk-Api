"""Train the credit risk model.

Run from the project folder:   python train.py

Steps:
  1. Load the data (downloads it the first time).
  2. Rename the columns to plain names and add extra features.
  3. Split into training data (80%) and test data (20%).
  4. Train a simple baseline model and the main LightGBM model.
  5. Compare them on the test data.
  6. Save the model so the API can use it.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import make_features

FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "credit.csv"
DATA_URL = "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/credit.csv"

# Only 22% of customers default, so a 50% cut-off would miss most of them.
# We decline anyone with a 30% or higher chance of defaulting.
THRESHOLD = 0.30

# ---------------------------------------------------------------- 1. Load
if not DATA_FILE.exists():
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
raw = pd.read_csv(DATA_FILE)

# ---------------------------------------------------------------- 2. Prepare
# Give the columns plain names. In the dataset, 1 = most recent month.
rename = {"LIMIT_BAL": "credit_limit", "EDUCATION": "education"}
for m in range(1, 7):
    rename[f"PAY_{m}"] = f"status_{m}"
    rename[f"BILL_AMT{m}"] = f"bill_{m}"
    rename[f"PAY_AMT{m}"] = f"paid_{m}"
df = raw.rename(columns=rename)

# Education codes 0, 5 and 6 aren't documented, so group them as 4 ("other").
df["education"] = df["education"].where(df["education"].isin([1, 2, 3, 4]), 4)

# We deliberately do NOT use sex, marital status or age as inputs,
# because lenders shouldn't make decisions based on them.
X = make_features(df[list(rename.values())])
y = df["default"]  # 1 = defaulted next month, 0 = paid

# ---------------------------------------------------------------- 3. Split
# stratify=y keeps the same 22% default rate in both parts.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------------- 4. Train
# Baseline: logistic regression, a simple straight-line model.
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
baseline.fit(X_train, y_train)

# Main model: LightGBM, which builds many small decision trees.
model = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80, random_state=42, verbose=-1)
model.fit(X_train, y_train)


# ---------------------------------------------------------------- 5. Evaluate
def score(m):
    prob = m.predict_proba(X_test)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, prob), 3),  # 0.5 = guessing, 1.0 = perfect
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),  # of those we decline, how many would default
        "recall": round(recall_score(y_test, pred), 3),  # of all defaulters, how many we catch
    }


results = {"lightgbm": score(model), "baseline_logistic_regression": score(baseline)}
print(json.dumps(results, indent=2))

# Fairness check: is the model equally accurate for men and women?
# (Sex is only used here to check, never as an input.)
prob = model.predict_proba(X_test)[:, 1]
sex = raw.loc[X_test.index, "SEX"]
fairness = {
    name: round(roc_auc_score(y_test[sex == code], prob[sex == code]), 3)
    for code, name in [(1, "male"), (2, "female")]
}
print("ROC AUC by group:", fairness)

# ---------------------------------------------------------------- 6. Save
model.booster_.save_model(str(FOLDER / "model.txt"))
info = {
    "dataset": "Default of Credit Card Clients (UCI, Taiwan 2005), 24,000 customers",
    "threshold": THRESHOLD,
    "test_results": results,
    "roc_auc_by_group": fairness,
}
(FOLDER / "model_info.json").write_text(json.dumps(info, indent=2))
print("Saved model.txt and model_info.json")
