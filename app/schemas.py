from datetime import date

from pydantic import BaseModel, Field


class ExpenseRecord(BaseModel):
    id: str
    user_id: str
    amount: float
    currency: str
    category: str
    description: str
    expense_date: date


class InsightsResponse(BaseModel):
    insights: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
