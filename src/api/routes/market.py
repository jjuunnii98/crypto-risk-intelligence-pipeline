from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/market", tags=["market"])

DATA_RAW_DIR = Path("data/raw")


def _get_latest_file(prefix: str) -> Path:
    if not DATA_RAW_DIR.exists():
        raise HTTPException(status_code=404, detail="raw data directory not found")

    files = sorted(DATA_RAW_DIR.glob(f"{prefix}_*.csv"))
    if not files:
        raise HTTPException(status_code=404, detail=f"no {prefix} files found")

    return files[-1]


@router.get("/ticker/latest")
def get_latest_ticker() -> dict[str, Any]:
    latest_file = _get_latest_file("ticker")
    df = pd.read_csv(latest_file)

    return {
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }


@router.get("/orderbook/latest")
def get_latest_orderbook() -> dict[str, Any]:
    latest_file = _get_latest_file("orderbook")
    df = pd.read_csv(latest_file)

    return {
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }


@router.get("/candles/latest")
def get_latest_candles() -> dict[str, Any]:
    latest_file = _get_latest_file("candles")
    df = pd.read_csv(latest_file)

    return {
        "data": df.to_dict(orient="records"),
        "source": str(latest_file),
    }