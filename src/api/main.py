from __future__ import annotations

from fastapi import FastAPI

from src.api.routes.alerts import router as alerts_router
from src.api.routes.market import router as market_router
from src.api.routes.risk import router as risk_router


app = FastAPI(
    title="Crypto Risk Intelligence API",
    description="API for crypto market risk evaluation, summaries, and alerts.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "Crypto Risk Intelligence API is running"
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok"
    }


app.include_router(risk_router)
app.include_router(market_router)
app.include_router(alerts_router)