"""Feature engineering: turning raw customer data into model inputs.

Used in two places:
  1. training/train.py, to prepare the training data.
  2. app/predict.py, to prepare each customer the API scores.

Using the SAME function in both places means the model always sees
data prepared in exactly the same way.
"""
import pandas as pd

# Month 1 is the most recent month, month 6 is the oldest.
MONTHS = [1, 2, 3, 4, 5, 6]

# Status codes in the data:
#   -2 = didn't use the card, -1 = paid in full, 0 = paid the minimum,
#    1 = payment 1 month late, 2 = 2 months late, ... up to 8.


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take the raw columns and add a few extra, more meaningful ones.

    Expects columns: credit_limit, education, status_1..6, bill_1..6, paid_1..6
    """
    X = df.copy()
    status = X[[f"status_{m}" for m in MONTHS]]

    # How many of the last 6 months was the customer late?
    X["months_late"] = (status > 0).sum(axis=1)

    # What was the worst delay? (0 if never late)
    X["worst_delay"] = status.max(axis=1).clip(lower=0)

    # What share of the credit limit is being used right now?
    X["credit_used"] = X["bill_1"] / X["credit_limit"]

    # Did they pay off last month's bill? (1.0 = paid it all, 0 = paid nothing)
    # If there was no bill, we count it as fully paid.
    last_bill = X["bill_2"].clip(lower=0)
    X["share_paid"] = (X["paid_1"] / last_bill.where(last_bill > 0)).fillna(1.0).clip(upper=1.0)

    return X


# Group the features into a few plain-English "reasons".
REASON_GROUPS = {
    "Late payments": ["months_late", "worst_delay"] + [f"status_{m}" for m in MONTHS],
    "Credit used": ["credit_used"] + [f"bill_{m}" for m in MONTHS],
    "Amount repaid": ["share_paid"] + [f"paid_{m}" for m in MONTHS],
    "Credit limit": ["credit_limit"],
    "Education": ["education"],
}
