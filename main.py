import json
import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama

from clients import fetch_recent_expenses
from prompts import SYSTEM_PROMPT_INSIGHTS
from schemas import ExpenseRecord, InsightsResponse
from security import oauth2_scheme, require_user

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "finagent-insights-service"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_request_log: dict[str, deque[float]] = defaultdict(deque)
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_ip = (
        request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
        or request.client.host
        or "anonymous"
    ).split(",")[0].strip()
    now = time.time()
    window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    limit = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    bucket = _request_log[client_ip]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    bucket.append(now)
    return await call_next(request)


@app.get("/api/v1/insights/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _fallback_response() -> InsightsResponse:
    return InsightsResponse(
        insights=["There was not enough recent expense data to generate strong insights."],
        suggestions=["Log more expenses over time to unlock better trend analysis."],
    )


def parse_markdown_sections(raw: str) -> InsightsResponse:
    insights: list[str] = []
    suggestions: list[str] = []
    current: list[str] | None = None

    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed == "**Spending Insights**":
            current = insights
            continue
        if trimmed == "**Spending Suggestions**":
            current = suggestions
            continue
        if trimmed.startswith("- ") and current is not None:
            current.append(trimmed[2:].strip())

    if not insights and not suggestions:
        return _fallback_response()

    return InsightsResponse(
        insights=insights or _fallback_response().insights,
        suggestions=suggestions or _fallback_response().suggestions,
    )


async def generate_insights(expenses: list[ExpenseRecord]) -> InsightsResponse:
    if not expenses:
        return InsightsResponse(
            insights=["No expenses were found for the last 30 days."],
            suggestions=["Log a few expenses to unlock personalized insights."],
        )

    serialized = json.dumps([expense.model_dump(mode="json") for expense in expenses], ensure_ascii=False)
    prompt = (
        f"{SYSTEM_PROMPT_INSIGHTS}\n\n"
        "Here is the user's expense history for the last 30 days as JSON:\n"
        f"{serialized}\n"
    )
    response = await llm.ainvoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return parse_markdown_sections(content)


@app.get("/api/v1/insights/summary", response_model=InsightsResponse)
async def summary(
    current_user: dict = Depends(require_user),
    token: str = Depends(oauth2_scheme),
) -> InsightsResponse:
    try:
        records = [ExpenseRecord.model_validate(item) for item in await fetch_recent_expenses(token, days=30)]
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Unable to fetch expenses for insights.") from exc

    try:
        return await generate_insights(records)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Unable to generate AI insights.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=os.getenv("APP_HOST", "0.0.0.0"), port=int(os.getenv("APP_PORT", "8003")), reload=True)
