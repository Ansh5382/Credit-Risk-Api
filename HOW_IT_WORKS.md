# How this project works (plain English)

## The idea in one paragraph

A bank wants to know, before giving someone more credit, how likely they are to **stop paying** their credit card bill next month. We take 24,000 real past customers where we already know who defaulted. We teach a model the patterns, like "people who paid late two months in a row often default". Then we put the model behind a web API, so any app can send a new customer's details and get back a risk score.

---

## The data

Each customer has:

| Column | Meaning |
|---|---|
| `credit_limit` | How much they're allowed to spend on the card |
| `education` | 1 graduate school, 2 university, 3 high school, 4 other |
| `status_1` … `status_6` | Repayment status for each of the last 6 months (1 = latest): -2 didn't use the card, -1 paid in full, 0 paid the minimum, 1–8 = that many months late |
| `bill_1` … `bill_6` | The statement amount each month |
| `paid_1` … `paid_6` | How much they paid each month |
| `default` | **What we predict:** 1 if they defaulted next month, 0 if not |

About **22%** of customers defaulted. That matters later.

---

## The files, in the order things happen

### 1. `features.py`: preparing the data

Models learn better when you hand them meaningful numbers. Besides the raw columns, we add four:

- **months_late**: how many of the 6 months they paid late
- **worst_delay**: their worst delay in months
- **credit_used**: this month's bill ÷ credit limit (0.9 means 90% of the limit used)
- **share_paid**: how much of last month's bill they paid off (1.0 = all of it)

Training and the API both use this file, so data is always prepared the same way.

### 2. `train.py`: teaching the model

1. **Load** the CSV and rename columns to plain names.
2. **Remove sensitive data.** We don't use sex, marital status or age as inputs, since lending decisions shouldn't depend on them.
3. **Split** into 80% training data and 20% test data. The model never sees the test data while learning, so the test score shows how it does on new customers.
4. **Train two models:**
   - **Logistic regression** is the *baseline*, a simple model to beat.
   - **LightGBM** is the main model. It builds hundreds of small decision trees, where each tree fixes some mistakes of the ones before.
5. **Measure** both on the test data (see below).
6. **Save** the model to `model.txt` and the results to `model_info.json`.

### 3. `predict.py`: scoring one customer

1. Loads the saved model once when the API starts.
2. Prepares the customer's data with `make_features()`.
3. Asks the model for a **probability of default**, e.g. 0.59 = 59%.
4. **Decision:** declines if the probability is 30% or more, otherwise approves.
5. **Reasons:** LightGBM can say how much each input pushed *this* customer's score up or down. These are called **SHAP values**. We add them up into five groups (Late payments, Credit used, Amount repaid, Credit limit, Education) and return the top 3.

### 4. `main.py`: the API

Built with **FastAPI**. It has four addresses (endpoints):

| Endpoint | What it does |
|---|---|
| `GET /` | The demo web page |
| `POST /predict` | Send a customer, get back a score |
| `GET /model-info` | The test results |
| `GET /health` | Says "ok" so hosting sites know it's running |

The `Customer` class describes what input is allowed. **Pydantic** checks it automatically. A negative credit limit or only 2 months of history gets a clear error (status 422) instead of a wrong answer.

### 5. `test_api.py`: automatic checks

Eight small tests, run with `pytest`. Examples: a customer who always pays in full is approved; a customer who's been late three months in a row is declined, with "Late payments" as the top reason.

### 6. `Dockerfile`: packaging it to run anywhere

A recipe that says "start from Python, install the libraries, copy the code, start the API". Hosting sites like Hugging Face follow this recipe to run the project.

---

## The results, explained

| Metric | LightGBM | Baseline | What it means |
|---|---|---|---|
| ROC AUC | **0.774** | 0.753 | How well the model ranks risky above safe customers. 0.5 is guessing, 1.0 is perfect. |
| Accuracy | **0.796** | 0.784 | Share of correct approve/decline decisions. |
| Precision | **0.537** | 0.510 | Of the people we decline, how many really would have defaulted. |
| Recall | 0.562 | 0.562 | Of all the people who defaulted, how many we caught. |

**Why a 30% cut-off and not 50%?** Only 22% of customers default. A model is rarely more than 50% sure about anyone, so a 50% cut-off would approve almost every defaulter. Lowering it to 30% catches more of them, at the cost of declining some good customers.

**Why not just use accuracy?** A model that approves *everyone* would score 78% accuracy, because 78% don't default, yet it would be useless. That's why we also look at ROC AUC, precision and recall.

**Fairness check:** ROC AUC is 0.78 for men and 0.77 for women, so the model works about equally well for both.

---

## Questions you might get in an interview

**Why LightGBM?**
It's one of the best model types for table data like this, it trains in seconds, and it can explain each prediction.

**Why have a baseline?**
Without one, you can't tell whether the complex model is worth it. Here it beats the baseline on ROC AUC, accuracy and precision.

**What is SHAP?**
A method that splits a prediction into how much each input added or removed, so you can tell a customer *why* they were declined.

**Why remove sex and age?**
Using them would be unfair and, in many countries, illegal for lending decisions. I still used sex *afterwards* to check the model works equally well for men and women.

**What would you improve?**
Train on newer data (this is from 2005), tune the model's settings, choose the cut-off based on the bank's real costs, and monitor the model once it's live.

**What's an API?**
A program on a server that other programs talk to. An app sends customer data to `/predict` and gets JSON back.
