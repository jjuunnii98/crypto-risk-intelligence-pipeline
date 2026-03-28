from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..data.database import SessionLocal
from ..data.db_models import CandleSnapshot, MarketSnapshot, NewsArticle, RiskSnapshot
from .formatters import parse_trigger_reasons


DATA_EVENTS_DIR = Path("data/events")
DATA_RAW_DIR = Path("data/raw")


def _rows_to_dataframe(rows: list[Any]) -> pd.DataFrame:
    """
    Convert SQLAlchemy ORM rows into a pandas DataFrame.
    """
    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append({column.name: getattr(row, column.name) for column in row.__table__.columns})
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_latest_csv(prefix: str, directory: Path) -> tuple[pd.DataFrame, Path | None]:
    """
    Load the latest CSV file matching the prefix from a directory.
    """
    if not directory.exists():
        return pd.DataFrame(), None

    files = sorted(directory.glob(f"{prefix}_*.csv"))
    if not files:
        return pd.DataFrame(), None

    latest_file = files[-1]
    return pd.read_csv(latest_file), latest_file


# =========================
# DB Loaders
# =========================
def _load_latest_risk_from_db(db: Session) -> pd.DataFrame:
    latest_created_at = db.execute(
        select(RiskSnapshot.created_at).order_by(desc(RiskSnapshot.created_at)).limit(1)
    ).scalar_one_or_none()

    if latest_created_at is None:
        return pd.DataFrame()

    rows = db.execute(
        select(RiskSnapshot)
        .where(RiskSnapshot.created_at == latest_created_at)
        .order_by(RiskSnapshot.asset)
    ).scalars().all()

    return _rows_to_dataframe(rows)


def _load_risk_history_from_db(db: Session, limit: int = 1000) -> pd.DataFrame:
    rows = db.execute(
        select(RiskSnapshot)
        .order_by(desc(RiskSnapshot.created_at), desc(RiskSnapshot.id))
        .limit(limit)
    ).scalars().all()

    df = _rows_to_dataframe(rows)
    if df.empty:
        return df

    return df.sort_values(["created_at", "asset"]).reset_index(drop=True)


def _load_latest_market_from_db(db: Session) -> pd.DataFrame:
    latest_collected_at = db.execute(
        select(MarketSnapshot.collected_at).order_by(desc(MarketSnapshot.collected_at)).limit(1)
    ).scalar_one_or_none()

    if latest_collected_at is None:
        return pd.DataFrame()

    rows = db.execute(
        select(MarketSnapshot)
        .where(MarketSnapshot.collected_at == latest_collected_at)
        .order_by(MarketSnapshot.market)
    ).scalars().all()

    return _rows_to_dataframe(rows)


def _load_market_history_from_db(db: Session, limit: int = 1500) -> pd.DataFrame:
    rows = db.execute(
        select(MarketSnapshot)
        .order_by(desc(MarketSnapshot.collected_at), desc(MarketSnapshot.id))
        .limit(limit)
    ).scalars().all()

    df = _rows_to_dataframe(rows)
    if df.empty:
        return df

    return df.sort_values(["collected_at", "market"]).reset_index(drop=True)


def _load_latest_news_from_db(db: Session, limit: int = 50) -> pd.DataFrame:
    rows = db.execute(
        select(NewsArticle)
        .order_by(desc(NewsArticle.published_at), desc(NewsArticle.id))
        .limit(limit)
    ).scalars().all()

    return _rows_to_dataframe(rows)


def _load_candle_history_from_db(db: Session, market: str, limit: int = 200) -> pd.DataFrame:
    rows = db.execute(
        select(CandleSnapshot)
        .where(CandleSnapshot.market == market)
        .where(CandleSnapshot.interval_type == "minute1")
        .order_by(desc(CandleSnapshot.candle_time_kst), desc(CandleSnapshot.id))
        .limit(limit)
    ).scalars().all()

    df = _rows_to_dataframe(rows)
    if df.empty:
        return df

    return df.sort_values("candle_time_kst").reset_index(drop=True)


# =========================
# Data Preparation Helpers
# =========================
def _convert_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    return result


def _convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_risk_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = _convert_datetime_columns(df, ["created_at"])
    result = _convert_numeric_columns(
        result,
        ["total_risk_score", "volatility_risk", "liquidity_risk", "sentiment_risk", "event_risk"],
    )

    if "should_alert" in result.columns:
        result["should_alert"] = result["should_alert"].astype(str).str.lower().isin(["true", "1"])

    if "trigger_reasons" in result.columns:
        result["trigger_reasons"] = result["trigger_reasons"].apply(parse_trigger_reasons)

    return result


def prepare_risk_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = _convert_datetime_columns(df, ["created_at"])
    result = _convert_numeric_columns(
        result,
        ["total_risk_score", "volatility_risk", "liquidity_risk", "sentiment_risk", "event_risk"],
    )

    result = result.sort_values(["asset", "created_at"]).reset_index(drop=True)
    if "total_risk_score" in result.columns:
        result["risk_change"] = result.groupby("asset")["total_risk_score"].diff()

    return result


def prepare_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = _convert_datetime_columns(df, ["collected_at"])
    result = _convert_numeric_columns(
        result,
        [
            "trade_price",
            "signed_change_rate",
            "total_ask_size",
            "total_bid_size",
            "ma_20",
            "ma_60",
            "rsi_14",
            "bollinger_upper_20",
            "bollinger_lower_20",
        ],
    )

    if "signed_change_rate" in result.columns:
        result["signed_change_rate_pct"] = result["signed_change_rate"] * 100

    return result


def prepare_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    return _convert_datetime_columns(df, ["published_at", "collected_at"])


def prepare_candle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = _convert_datetime_columns(df, ["candle_time_kst", "candle_time_utc", "created_at"])
    result = _convert_numeric_columns(
        result,
        [
            "opening_price",
            "high_price",
            "low_price",
            "trade_price",
            "candle_acc_trade_volume",
            "candle_acc_trade_price",
        ],
    )

    result = result.dropna(
        subset=["candle_time_kst", "opening_price", "high_price", "low_price", "trade_price"]
    ).reset_index(drop=True)

    return result


# =========================
# Public Dashboard Loaders
# =========================
@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    db = SessionLocal()
    try:
        risk_df = _load_latest_risk_from_db(db)
        risk_history_df = _load_risk_history_from_db(db)
        market_df = _load_latest_market_from_db(db)
        market_history_df = _load_market_history_from_db(db)
        news_df = _load_latest_news_from_db(db)
    finally:
        db.close()

    return {
        "risk_df": prepare_risk_dataframe(risk_df),
        "risk_history_df": prepare_risk_history_dataframe(risk_history_df),
        "market_df": prepare_market_dataframe(market_df),
        "market_history_df": prepare_market_dataframe(market_history_df),
        "news_df": prepare_news_dataframe(news_df),
    }


@st.cache_data(show_spinner=False)
def load_candle_history(market: str, limit: int = 200) -> pd.DataFrame:
    db = SessionLocal()
    try:
        candle_df = _load_candle_history_from_db(db, market=market, limit=limit)
    finally:
        db.close()

    return prepare_candle_dataframe(candle_df)