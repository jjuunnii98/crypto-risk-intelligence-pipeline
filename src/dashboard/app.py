

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Crypto Risk Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


DATA_EVENTS_DIR = Path("data/events")
DATA_RAW_DIR = Path("data/raw")


@st.cache_data(show_spinner=False)
def load_latest_csv(prefix: str, directory: Path) -> tuple[pd.DataFrame, Path | None]:
    """
    Load the latest CSV file matching the given prefix from a directory.
    """
    if not directory.exists():
        return pd.DataFrame(), None

    files = sorted(directory.glob(f"{prefix}_*.csv"))
    if not files:
        return pd.DataFrame(), None

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return df, latest_file


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    """
    Load latest risk event data and latest raw market/news data.
    """
    risk_df, risk_file = load_latest_csv("risk_events", DATA_EVENTS_DIR)
    ticker_df, ticker_file = load_latest_csv("ticker", DATA_RAW_DIR)
    news_df, news_file = load_latest_csv("news", DATA_RAW_DIR)

    return {
        "risk_df": risk_df,
        "risk_file": risk_file,
        "ticker_df": ticker_df,
        "ticker_file": ticker_file,
        "news_df": news_df,
        "news_file": news_file,
    }


@st.cache_data(show_spinner=False)
def prepare_risk_dataframe(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare risk dataframe for dashboard display.
    """
    if risk_df.empty:
        return risk_df

    df = risk_df.copy()

    if "latest_timestamp" in df.columns:
        df["latest_timestamp"] = pd.to_datetime(df["latest_timestamp"], errors="coerce")

    if "total_risk_score" in df.columns:
        df["total_risk_score"] = pd.to_numeric(df["total_risk_score"], errors="coerce").round(4)

    for column in [
        "volatility_risk",
        "liquidity_risk",
        "sentiment_risk",
        "event_risk",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(4)

    return df


@st.cache_data(show_spinner=False)
def prepare_ticker_dataframe(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ticker dataframe for dashboard display.
    """
    if ticker_df.empty:
        return ticker_df

    df = ticker_df.copy()

    if "trade_price" in df.columns:
        df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")
        df["trade_price_formatted"] = df["trade_price"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    if "signed_change_rate" in df.columns:
        df["signed_change_rate_pct"] = (
            pd.to_numeric(df["signed_change_rate"], errors="coerce") * 100
        ).round(2)

    return df


def render_header(risk_file: Path | None, ticker_file: Path | None, news_file: Path | None) -> None:
    st.title("Crypto Risk Intelligence Dashboard")
    st.caption(
        "Upbit market data + Google/Naver News based risk monitoring dashboard"
    )

    with st.expander("Data Sources", expanded=False):
        st.write(
            {
                "risk_events": str(risk_file) if risk_file else None,
                "ticker": str(ticker_file) if ticker_file else None,
                "news": str(news_file) if news_file else None,
            }
        )


def render_kpis(risk_df: pd.DataFrame) -> None:
    if risk_df.empty:
        st.warning("No risk evaluation data found. Run the data pipeline first.")
        return

    total_assets = int(len(risk_df))
    alert_count = int(risk_df["should_alert"].sum()) if "should_alert" in risk_df.columns else 0
    avg_total_risk = (
        float(risk_df["total_risk_score"].mean()) if "total_risk_score" in risk_df.columns else 0.0
    )

    highest_asset = "-"
    highest_score = 0.0
    if "total_risk_score" in risk_df.columns and "asset" in risk_df.columns and not risk_df.empty:
        highest_row = risk_df.sort_values("total_risk_score", ascending=False).iloc[0]
        highest_asset = str(highest_row["asset"])
        highest_score = float(highest_row["total_risk_score"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracked Assets", total_assets)
    col2.metric("Alert Count", alert_count)
    col3.metric("Average Total Risk", f"{avg_total_risk:.4f}")
    col4.metric("Highest Risk Asset", f"{highest_asset} ({highest_score:.4f})")


def render_asset_selector(risk_df: pd.DataFrame) -> str | None:
    if risk_df.empty or "asset" not in risk_df.columns:
        return None

    assets = sorted(risk_df["asset"].dropna().astype(str).unique().tolist())
    if not assets:
        return None

    return st.selectbox("Select Asset", assets, index=0)


def render_asset_summary(selected_asset: str, risk_df: pd.DataFrame, ticker_df: pd.DataFrame) -> None:
    asset_risk = risk_df.loc[risk_df["asset"] == selected_asset].copy()
    asset_ticker = ticker_df.loc[ticker_df["market"] == selected_asset].copy() if not ticker_df.empty else pd.DataFrame()

    if asset_risk.empty:
        st.info(f"No risk data found for {selected_asset}.")
        return

    risk_row = asset_risk.iloc[0]

    st.subheader(f"Asset Summary: {selected_asset}")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Risk Level", str(risk_row.get("risk_level", "-")))
    summary_col2.metric("Should Alert", str(risk_row.get("should_alert", False)))
    summary_col3.metric("Total Risk Score", f"{float(risk_row.get('total_risk_score', 0.0)):.4f}")

    if not asset_ticker.empty:
        ticker_row = asset_ticker.iloc[0]
        market_col1, market_col2 = st.columns(2)
        market_col1.metric(
            "Trade Price",
            str(ticker_row.get("trade_price_formatted", "-")),
        )
        change_pct = ticker_row.get("signed_change_rate_pct")
        market_col2.metric(
            "Signed Change Rate (%)",
            f"{change_pct:.2f}%" if pd.notna(change_pct) else "-",
        )

    component_table = pd.DataFrame(
        {
            "component": [
                "volatility_risk",
                "liquidity_risk",
                "sentiment_risk",
                "event_risk",
            ],
            "score": [
                float(risk_row.get("volatility_risk", 0.0)),
                float(risk_row.get("liquidity_risk", 0.0)),
                float(risk_row.get("sentiment_risk", 0.0)),
                float(risk_row.get("event_risk", 0.0)),
            ],
        }
    )

    st.markdown("### Component Risk Scores")
    st.bar_chart(component_table.set_index("component"))

    trigger_reasons = risk_row.get("trigger_reasons", "[]")
    st.markdown("### Trigger Reasons")
    st.code(str(trigger_reasons))


def render_risk_table(risk_df: pd.DataFrame) -> None:
    st.subheader("Risk Evaluation Table")
    if risk_df.empty:
        st.info("No risk evaluation data available.")
        return

    display_columns = [
        "asset",
        "latest_timestamp",
        "volatility_risk",
        "liquidity_risk",
        "sentiment_risk",
        "event_risk",
        "total_risk_score",
        "risk_level",
        "should_alert",
        "news_count",
        "negative_news_count",
        "event_news_count",
    ]
    available_columns = [col for col in display_columns if col in risk_df.columns]
    st.dataframe(risk_df[available_columns], use_container_width=True)


def render_news_section(news_df: pd.DataFrame) -> None:
    st.subheader("Latest News Snapshot")
    if news_df.empty:
        st.info("No news data available.")
        return

    display_columns = [col for col in ["keyword", "title", "source", "published_at"] if col in news_df.columns]
    st.dataframe(news_df[display_columns].head(20), use_container_width=True)


def main() -> None:
    dashboard_data = load_dashboard_data()

    risk_df = prepare_risk_dataframe(dashboard_data["risk_df"])
    ticker_df = prepare_ticker_dataframe(dashboard_data["ticker_df"])
    news_df = dashboard_data["news_df"]

    render_header(
        risk_file=dashboard_data["risk_file"],
        ticker_file=dashboard_data["ticker_file"],
        news_file=dashboard_data["news_file"],
    )
    render_kpis(risk_df)

    selected_asset = render_asset_selector(risk_df)
    if selected_asset:
        render_asset_summary(selected_asset, risk_df, ticker_df)

    st.divider()
    render_risk_table(risk_df)

    st.divider()
    render_news_section(news_df)


if __name__ == "__main__":
    main()