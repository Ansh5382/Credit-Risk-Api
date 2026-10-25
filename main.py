"""The web API, built with FastAPI.

Start it with:   uvicorn main:app --reload
Then open http://localhost:8000 for the demo page,
or http://localhost:8000/docs to try the API directly.
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predict import model_info, predict

app = FastAPI(title="Credit Default Risk API", description="Predicts the chance a credit card customer defaults next month.")

EDUCATION_CODES = {"graduate_school": 1, "university": 2, "high_school": 3, "other": 4}


class Customer(BaseModel):
    """What the API expects. Every list holds 6 months, most recent month first.

    Pydantic checks these rules automatically and returns a clear error if
    something is wrong (for example a negative credit limit).
    """

    credit_limit: float = Field(gt=0, examples=[50000])
    education: Literal["graduate_school", "university", "high_school", "other"] = Field(examples=["university"])
    status: list[int] = Field(
        min_length=6, max_length=6, examples=[[2, 2, 0, 0, 0, 0]],
        description="-2 didn't use card, -1 paid in full, 0 paid minimum, 1-8 months late",
    )
    bills: list[float] = Field(min_length=6, max_length=6, examples=[[46990, 48233, 49291, 28314, 28959, 29547]])
    paid: list[float] = Field(min_length=6, max_length=6, examples=[[0, 2019, 1200, 1100, 1069, 1000]])


@app.get("/", include_in_schema=False)
def home():
    """Serve the demo web page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    """Used by hosting sites to check the API is running."""
    return {"status": "ok"}


@app.get("/model-info")
def info():
    """How well the model did on test data."""
    return model_info


@app.post("/predict")
def predict_customer(customer: Customer):
    """Score one customer."""
    # Turn the lists into the column names the model was trained on.
    row = {"credit_limit": customer.credit_limit, "education": EDUCATION_CODES[customer.education]}
    for i in range(6):
        row[f"status_{i + 1}"] = customer.status[i]
        row[f"bill_{i + 1}"] = customer.bills[i]
        row[f"paid_{i + 1}"] = customer.paid[i]
    return predict(row)
