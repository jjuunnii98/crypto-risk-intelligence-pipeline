from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.services.risk_service import evaluate_risk_payload

router = APIRouter(prefix="/risk", tags=["risk"])

DATA_EVENTS_DIR = Path("data/events")


def _get_latest_event_file() -> Path:
    """
    Return the most recent risk_events CSV file.
    """
    if not DATA_EVENTS_DIR.exists():
        raise HTTPException(status_code=404, detail="events directory not found")

    files = sorted(DATA_EVENTS_DIR.glob("risk_events_*.csv"))
    if not files:
        raise HTTPException(status_code=404, detail="no risk event files found")

    return files[-1]


@router.get("/latest")
def get_latest_risk() -> dict[str, Any]:
    """
    Get latest saved risk evaluation results.
    """
    latest_file = _get_latest_event_file()

    df = pd.read_csv(latest_file)
    if df.empty:
        return {"data": [], "source": str(latest_file)}

    return {
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }


@router.get("/asset/{asset}")
def get_asset_risk(asset: str) -> dict[str, Any]:
    """
    Get latest risk for a specific asset (e.g., KRW-BTC).
    """
    latest_file = _get_latest_event_file()

    df = pd.read_csv(latest_file)
    if df.empty:
        raise HTTPException(status_code=404, detail="no data available")

    filtered = df.loc[df["asset"] == asset]
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"asset {asset} not found")

    return filtered.to_dict(orient="records")[0]


@router.post("/evaluate")
def evaluate_risk(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate risk in real-time using input payload.

    Example payload:
    {
        "asset": "KRW-BTC",
        "volatility_risk": 0.3,
        "liquidity_risk": 0.5,
        "sentiment_risk": 0.2,
        "event_risk": 0.1
    }
    """
    try:
        result = evaluate_risk_payload(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.get("/summary")
def get_risk_summary(
    level: str | None = Query(default=None, description="Filter by risk level")
) -> dict[str, Any]:
    """
    Get summarized risk overview.

    Optional:
    - level: normal / caution / warning / critical
    """
    latest_file = _get_latest_event_file()

    df = pd.read_csv(latest_file)
    if df.empty:
        return {"summary": {}, "data": []}

    if level:
        df = df.loc[df["risk_level"] == level]

    summary = {
        "total_assets": int(len(df)),
        "risk_distribution": df["risk_level"].value_counts().to_dict(),
        "alerts": int(df["should_alert"].sum()) if "should_alert" in df.columns else 0,
    }

    return {
        "summary": summary,
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }