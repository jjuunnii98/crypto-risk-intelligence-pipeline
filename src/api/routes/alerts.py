from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/alerts", tags=["alerts"])

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
def get_latest_alerts() -> dict[str, Any]:
    """
    Get latest alertable events from the newest risk_events file.
    """
    latest_file = _get_latest_event_file()

    df = pd.read_csv(latest_file)
    if df.empty:
        return {"data": [], "source": str(latest_file)}

    if "should_alert" in df.columns:
        df = df.loc[df["should_alert"] == True]

    return {
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }


@router.get("/summary")
def get_alert_summary(
    only_alerts: bool = Query(default=True, description="Return only rows where should_alert is true")
) -> dict[str, Any]:
    """
    Get summary of latest alert results.
    """
    latest_file = _get_latest_event_file()

    df = pd.read_csv(latest_file)
    if df.empty:
        return {"summary": {}, "data": [], "source": str(latest_file)}

    if only_alerts and "should_alert" in df.columns:
        df = df.loc[df["should_alert"] == True]

    summary = {
        "total_rows": int(len(df)),
        "risk_distribution": df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {},
        "alert_count": int(df["should_alert"].sum()) if "should_alert" in df.columns else 0,
    }

    return {
        "summary": summary,
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }