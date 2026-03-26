from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import desc, select
from streamlit import column_config

from src.data.database import SessionLocal
from src.data.db_models import MarketSnapshot, NewsArticle, RiskSnapshot
from src.llm.risk_explainer import filter_asset_news, generate_hybrid_risk_insight


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
    Load dashboard data from SQLite first.
    Falls back to CSV files if DB tables are empty.
    """
    risk_df, ticker_df, news_df, risk_history_df = _load_dashboard_data_from_db()

    risk_source = "sqlite:risk_snapshots"
    ticker_source = "sqlite:market_snapshots"
    news_source = "sqlite:news_articles"
    risk_history_source = "sqlite:risk_snapshots(history)"

    if risk_df.empty:
        risk_df, risk_file = load_latest_csv("risk_events", DATA_EVENTS_DIR)
        risk_source = str(risk_file) if risk_file else None
    if ticker_df.empty:
        ticker_df, ticker_file = load_latest_csv("ticker", DATA_RAW_DIR)
        ticker_source = str(ticker_file) if ticker_file else None
    if news_df.empty:
        news_df, news_file = load_latest_csv("news", DATA_RAW_DIR)
        news_source = str(news_file) if news_file else None

    return {
        "risk_df": risk_df,
        "risk_source": risk_source,
        "ticker_df": ticker_df,
        "ticker_source": ticker_source,
        "news_df": news_df,
        "news_source": news_source,
        "risk_history_df": risk_history_df,
        "risk_history_source": risk_history_source,
    }


@st.cache_data(show_spinner=False)
def _load_dashboard_data_from_db() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load latest risk, market, news, and risk history data from SQLite.
    """
    db = SessionLocal()
    try:
        risk_df = _load_latest_risk_from_db(db)
        ticker_df = _load_latest_market_from_db(db)
        news_df = _load_latest_news_from_db(db)
        risk_history_df = _load_risk_history_from_db(db)
    finally:
        db.close()

    return risk_df, ticker_df, news_df, risk_history_df



def _rows_to_dataframe(rows: list[Any], columns: list[str] | None = None) -> pd.DataFrame:
    """
    Convert SQLAlchemy row objects to pandas DataFrame.
    """
    if not rows:
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)

    records = []
    for row in rows:
        records.append({column.name: getattr(row, column.name) for column in row.__table__.columns})

    return pd.DataFrame(records)


def _load_latest_risk_from_db(db) -> pd.DataFrame:
    """
    Load only the most recent risk snapshot timestamp from DB.
    """
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


def _load_risk_history_from_db(db, limit: int = 500) -> pd.DataFrame:
    """
    Load recent historical risk snapshots from DB for time-series visualization.
    """
    rows = db.execute(
        select(RiskSnapshot)
        .order_by(desc(RiskSnapshot.created_at), desc(RiskSnapshot.id))
        .limit(limit)
    ).scalars().all()

    history_df = _rows_to_dataframe(rows)
    if history_df.empty:
        return history_df

    return history_df.sort_values(["created_at", "asset"]).reset_index(drop=True)


def _load_latest_market_from_db(db) -> pd.DataFrame:
    """
    Load the most recent market snapshot per asset from DB.
    """
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


def _load_latest_news_from_db(db, limit: int = 50) -> pd.DataFrame:
    """
    Load recent news rows from DB.
    """
    rows = db.execute(
        select(NewsArticle)
        .order_by(desc(NewsArticle.published_at), desc(NewsArticle.id))
        .limit(limit)
    ).scalars().all()

    return _rows_to_dataframe(rows)


@st.cache_data(show_spinner=False)
def prepare_risk_dataframe(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare risk dataframe for dashboard display.
    """
    if risk_df.empty:
        return risk_df

    df = risk_df.copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "latest_timestamp" in df.columns:
        df["latest_timestamp"] = pd.to_datetime(df["latest_timestamp"], errors="coerce", utc=True)

    numeric_columns = [
        "total_risk_score",
        "volatility_risk",
        "liquidity_risk",
        "sentiment_risk",
        "event_risk",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(4)

    if "should_alert" in df.columns:
        df["should_alert"] = df["should_alert"].astype(str).str.lower().isin(["true", "1"])

    if "trigger_reasons" in df.columns:
        df["trigger_reasons"] = df["trigger_reasons"].apply(_parse_trigger_reasons)

    return df


@st.cache_data(show_spinner=False)
def prepare_risk_history_dataframe(risk_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare historical risk dataframe for dashboard time-series charts.
    """
    if risk_history_df.empty:
        return risk_history_df

    df = risk_history_df.copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    for column in [
        "total_risk_score",
        "volatility_risk",
        "liquidity_risk",
        "sentiment_risk",
        "event_risk",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["created_at", "asset", "total_risk_score"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def prepare_ticker_dataframe(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ticker dataframe for dashboard display.
    """
    if ticker_df.empty:
        return ticker_df

    df = ticker_df.copy()

    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)

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


@st.cache_data(show_spinner=False)
def prepare_news_dataframe(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare news dataframe for dashboard display.
    """
    if news_df.empty:
        return news_df

    df = news_df.copy()

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)

    return df


def _parse_trigger_reasons(value: Any) -> list[str]:
    """
    Parse stored trigger reasons from JSON/string/list.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [text]




def render_component_risk_chart(component_table: pd.DataFrame) -> None:
    """
    Render component risk scores with horizontal x-axis labels.
    """
    chart = (
        alt.Chart(component_table)
        .mark_bar()
        .encode(
            x=alt.X(
                "component:N",
                title="Risk Component",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "score:Q",
                title="Score",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def render_header(
    risk_source: str | None,
    ticker_source: str | None,
    news_source: str | None,
    risk_history_source: str | None,
) -> None:
    st.title("Crypto Risk Intelligence Dashboard")
    st.caption("Upbit market data + Google/Naver News based risk monitoring dashboard")

    with st.expander("Data Sources", expanded=False):
        st.write(
            {
                "risk": risk_source,
                "market": ticker_source,
                "news": news_source,
                "risk_history": risk_history_source,
            }
        )


def render_kpis(risk_df: pd.DataFrame) -> None:
    if risk_df.empty:
        st.warning("No risk evaluation data found. Run the realtime pipeline first.")
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


def render_alert_banner(risk_df: pd.DataFrame) -> None:
    """
    Show an alert banner when any tracked asset requires action.
    """
    if risk_df.empty or "should_alert" not in risk_df.columns:
        return

    alert_df = risk_df.loc[risk_df["should_alert"] == True].copy()
    if alert_df.empty:
        st.success("현재 즉시 대응이 필요한 자산은 없습니다. 실시간 모니터링 상태는 안정적입니다.")
        return

    alert_assets = ", ".join(alert_df["asset"].astype(str).tolist())
    st.error(f"즉시 확인이 필요한 자산이 감지되었습니다: {alert_assets}")


def render_total_risk_timeseries(risk_history_df: pd.DataFrame) -> None:
    """
    Render historical total risk score chart.
    """
    st.subheader("Historical Total Risk Trend")
    if risk_history_df.empty:
        st.info("No historical risk snapshots available yet.")
        return

    chart = (
        alt.Chart(risk_history_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Timestamp"),
            y=alt.Y("total_risk_score:Q", title="Total Risk Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("asset:N", title="Asset"),
            tooltip=[
                alt.Tooltip("created_at:T", title="Timestamp"),
                alt.Tooltip("asset:N", title="Asset"),
                alt.Tooltip("total_risk_score:Q", title="Total Risk", format=".4f"),
                alt.Tooltip("risk_level:N", title="Risk Level"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def render_component_history_chart(risk_history_df: pd.DataFrame, selected_asset: str | None) -> None:
    """
    Render component risk history for the selected asset.
    """
    st.subheader("Historical Component Risk Trend")
    if risk_history_df.empty or not selected_asset:
        st.info("Select an asset to view component-level history.")
        return

    asset_history = risk_history_df.loc[risk_history_df["asset"] == selected_asset].copy()
    if asset_history.empty:
        st.info("No historical component data available for the selected asset.")
        return

    component_columns = [
        "volatility_risk",
        "liquidity_risk",
        "sentiment_risk",
        "event_risk",
    ]
    available_components = [column for column in component_columns if column in asset_history.columns]
    if not available_components:
        st.info("No component history data available.")
        return

    long_df = asset_history.melt(
        id_vars=["created_at", "asset"],
        value_vars=available_components,
        var_name="component",
        value_name="score",
    )

    chart = (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Timestamp"),
            y=alt.Y("score:Q", title="Component Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("component:N", title="Component"),
            tooltip=[
                alt.Tooltip("created_at:T", title="Timestamp"),
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def render_asset_selector(risk_df: pd.DataFrame) -> str | None:
    if risk_df.empty or "asset" not in risk_df.columns:
        return None

    assets = sorted(risk_df["asset"].dropna().astype(str).unique().tolist())
    if not assets:
        return None

    return st.selectbox("Select Asset", assets, index=0)


def render_asset_summary(selected_asset: str, risk_df: pd.DataFrame, ticker_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
    asset_risk = risk_df.loc[risk_df["asset"] == selected_asset].copy()
    asset_ticker = ticker_df.loc[ticker_df.get("market", pd.Series(dtype=str)) == selected_asset].copy() if not ticker_df.empty else pd.DataFrame()
    asset_news = filter_asset_news(news_df, selected_asset)

    if asset_risk.empty:
        st.info(f"No risk data found for {selected_asset}.")
        return

    risk_row = asset_risk.iloc[0]
    insight = generate_hybrid_risk_insight(risk_row, asset_news)

    st.subheader(f"Asset Summary: {selected_asset}")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Risk Level", str(risk_row.get("risk_level", "-")))
    summary_col2.metric("Should Alert", str(risk_row.get("should_alert", False)))
    summary_col3.metric("Total Risk Score", f"{float(risk_row.get('total_risk_score', 0.0)):.4f}")

    if not asset_ticker.empty:
        ticker_row = asset_ticker.iloc[0]
        market_col1, market_col2 = st.columns(2)
        market_col1.metric("Trade Price", str(ticker_row.get("trade_price_formatted", "-")))
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
    render_component_risk_chart(component_table)

    trigger_reasons = risk_row.get("trigger_reasons", [])
    st.markdown("### Trigger Reasons")
    if trigger_reasons:
        for reason in trigger_reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No trigger reasons")

    st.markdown("### Hybrid AI Risk Insight")
    st.info(insight["summary"])
    st.write(insight["opinion"])
    st.caption(insight["news_context"])

    st.markdown("### AI Solution / Operator Actions")
    for action in insight["operator_actions"]:
        st.write(f"- {action}")

    st.markdown("### Recent Asset News")
    if asset_news.empty:
        st.write("- No recent asset-specific news found")
    else:
        news_display = asset_news.copy()
        if "link" in news_display.columns:
            news_display["open_link"] = news_display["link"]
        display_columns = [
            column for column in ["published_at", "source", "title", "open_link"] if column in news_display.columns
        ]
        st.dataframe(
            news_display[display_columns].head(10),
            use_container_width=True,
            column_config={
                "open_link": column_config.LinkColumn(
                    "Open Article",
                    display_text="기사 열기",
                )
            },
            hide_index=True,
        )


def render_risk_table(risk_df: pd.DataFrame) -> None:
    st.subheader("Risk Evaluation Table")
    if risk_df.empty:
        st.info("No risk evaluation data available.")
        return

    display_columns = [
        "asset",
        "created_at",
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
    available_columns = [column for column in display_columns if column in risk_df.columns]
    st.dataframe(risk_df[available_columns], use_container_width=True)


def render_news_section(news_df: pd.DataFrame) -> None:
    st.subheader("Latest News Snapshot")
    if news_df.empty:
        st.info("No news data available.")
        return

    display_df = news_df.copy()
    if "link" in display_df.columns:
        display_df["open_link"] = display_df["link"]

    display_columns = [
        column for column in ["keyword", "title", "source", "published_at", "open_link"] if column in display_df.columns
    ]
    st.dataframe(
        display_df[display_columns].head(20),
        use_container_width=True,
        column_config={
            "open_link": column_config.LinkColumn(
                "Open Article",
                display_text="기사 열기",
            )
        },
        hide_index=True,
    )


def main() -> None:
    st.sidebar.title("Controls")
    refresh_requested = st.sidebar.button("Refresh Data")
    if refresh_requested:
        st.cache_data.clear()

    dashboard_data = load_dashboard_data()

    risk_df = prepare_risk_dataframe(dashboard_data["risk_df"])
    risk_history_df = prepare_risk_history_dataframe(dashboard_data["risk_history_df"])
    ticker_df = prepare_ticker_dataframe(dashboard_data["ticker_df"])
    news_df = prepare_news_dataframe(dashboard_data["news_df"])

    render_header(
        risk_source=dashboard_data["risk_source"],
        ticker_source=dashboard_data["ticker_source"],
        news_source=dashboard_data["news_source"],
        risk_history_source=dashboard_data["risk_history_source"],
    )
    render_alert_banner(risk_df)
    render_kpis(risk_df)

    selected_asset = render_asset_selector(risk_df)

    overview_tab, asset_tab, news_tab = st.tabs(["Overview", "Asset Drilldown", "News & Table"])

    with overview_tab:
        render_total_risk_timeseries(risk_history_df)
        st.divider()
        render_risk_table(risk_df)

    with asset_tab:
        if selected_asset:
            render_asset_summary(selected_asset, risk_df, ticker_df, news_df)
            st.divider()
            render_component_history_chart(risk_history_df, selected_asset)
        else:
            st.info("No asset available for drilldown.")

    with news_tab:
        render_news_section(news_df)


if __name__ == "__main__":
    main()