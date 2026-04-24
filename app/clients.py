import os

import httpx


async def fetch_recent_expenses(token: str, days: int = 30) -> list[dict]:
    async with httpx.AsyncClient(base_url=os.getenv("EXPENSE_SERVICE_BASE_URL", "http://localhost:8002"), timeout=20.0) as client:
        response = await client.get(
            "/api/v1/expenses",
            params={"days": days},
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()
