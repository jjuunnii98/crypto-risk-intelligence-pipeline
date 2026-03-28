from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit import column_config

from .formatters import format_pct, format_price
from ..llm.risk_explainer import filter_asset_news, generate_hybrid_risk_insight


NEWS_DISPLAY_COLUMNS = ["keyword", "title", "source", "published_at", "open_link"]
RISK_TABLE_COLUMNS = [
    "asset",
    "risk_level",
    "total_risk_score",
    "volatility_risk",
    "liquidity_risk",
    "sentiment_risk",
    "event_risk",
    "should_alert",
    "created_at",
]


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return a dataframe column when it exists, otherwise an empty series.
    """
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=str)
    return df[column]


def _get_latest_highest_risk_asset(risk_df: pd.DataFrame) -> tuple[str, float]:
    """
    Return the asset with the highest total risk score.
    """
    if risk_df.empty or "total_risk_score" not in risk_df.columns or "asset" not in risk_df.columns:
        return "-", 0.0

    working_df = risk_df.dropna(subset=["total_risk_score"]).copy()
    if working_df.empty:
        return "-", 0.0

    row = working_df.sort_values("total_risk_score", ascending=False).iloc[0]
    return str(row["asset"]), float(row["total_risk_score"])


def render_header() -> None:
    st.title("Crypto Risk Intelligence Dashboard")
    st.caption("Upbit market data + Google/Naver News based risk monitoring dashboard")


def render_alert_banner(risk_df: pd.DataFrame) -> None:
    if risk_df.empty or "should_alert" not in risk_df.columns:
        return

    alert_df = risk_df.loc[risk_df["should_alert"] == True].copy()
    if alert_df.empty:
        st.success("현재 즉시 대응이 필요한 자산은 없습니다. 실시간 모니터링 상태는 안정적입니다.")
        return

    alert_assets = ", ".join(alert_df["asset"].astype(str).tolist())
    st.error(f"즉시 확인이 필요한 자산이 감지되었습니다: {alert_assets}")


def render_kpis(risk_df: pd.DataFrame, risk_history_df: pd.DataFrame) -> None:
    if risk_df.empty:
        st.warning("No risk evaluation data found.")
        return

    total_assets = int(len(risk_df))
    alert_count = int(risk_df["should_alert"].sum()) if "should_alert" in risk_df.columns else 0
    avg_total_risk = float(risk_df["total_risk_score"].mean()) if "total_risk_score" in risk_df.columns else 0.0
    snapshot_count = int(len(risk_history_df)) if not risk_history_df.empty else 0
    highest_asset, highest_score = _get_latest_highest_risk_asset(risk_df)

    cols = st.columns(5)
    cols[0].metric("Tracked Assets", total_assets)
    cols[1].metric("Alert Count", alert_count)
    cols[2].metric("Average Total Risk", f"{avg_total_risk:.4f}")
    cols[3].metric("Highest Risk Asset", f"{highest_asset} ({highest_score:.4f})")
    cols[4].metric("Snapshots Stored", snapshot_count)


def render_asset_summary(
    selected_asset: str,
    risk_df: pd.DataFrame,
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> None:
    asset_risk = risk_df.loc[_safe_series(risk_df, "asset") == selected_asset].copy()
    asset_market = market_df.loc[_safe_series(market_df, "market") == selected_asset].copy()
    asset_news = filter_asset_news(news_df, selected_asset)

    if asset_risk.empty:
        st.info(f"No risk data found for {selected_asset}.")
        return

    risk_row = asset_risk.iloc[0]
    insight = generate_hybrid_risk_insight(risk_row, asset_news)

    st.subheader(f"Asset Summary: {selected_asset}")

    cols = st.columns(3)
    cols[0].metric("Risk Level", str(risk_row.get("risk_level", "-")))
    cols[1].metric("Should Alert", str(risk_row.get("should_alert", False)))
    cols[2].metric("Total Risk Score", f"{float(risk_row.get('total_risk_score', 0.0)):.4f}")

    if not asset_market.empty:
        market_row = asset_market.iloc[0]
        mcols = st.columns(2)
        mcols[0].metric("Trade Price", format_price(market_row.get("trade_price")))
        mcols[1].metric("Signed Change Rate (%)", format_pct(market_row.get("signed_change_rate_pct")))

    st.markdown("### Hybrid AI Risk Insight")
    st.info(insight.get("summary", "No summary available."))
    st.write(insight.get("opinion", "No opinion available."))

    news_context = insight.get("news_context")
    if news_context:
        st.caption(news_context)

    st.markdown("### AI Solution / Operator Actions")
    operator_actions = insight.get("operator_actions", [])
    if operator_actions:
        for action in operator_actions:
            st.write(f"- {action}")
    else:
        st.write("- No operator actions available.")


def render_risk_table_section(risk_df: pd.DataFrame) -> None:
    """
    Render a compact product-style risk monitoring table.
    """
    st.subheader("Latest Risk Monitor")

    if risk_df.empty:
        st.info("No risk evaluation data available.")
        return

    display_df = risk_df.copy()
    display_columns = [column for column in RISK_TABLE_COLUMNS if column in display_df.columns]
    if not display_columns:
        st.info("No risk columns available for display.")
        return

    if "total_risk_score" in display_df.columns:
        display_df = display_df.sort_values("total_risk_score", ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )


def render_news_section(news_df: pd.DataFrame) -> None:
    st.subheader("Latest News Snapshot")
    if news_df.empty:
        st.info("No news data available.")
        return

    display_df = news_df.copy()
    if "link" in display_df.columns:
        display_df["open_link"] = display_df["link"]

    if "published_at" in display_df.columns:
        display_df = display_df.sort_values("published_at", ascending=False).reset_index(drop=True)

    top_col1, top_col2, top_col3 = st.columns(3)
    top_col1.metric("Articles", int(len(display_df)))
    top_col2.metric(
        "Sources",
        int(display_df["source"].nunique()) if "source" in display_df.columns else 0,
    )
    top_col3.metric(
        "Keywords",
        int(display_df["keyword"].nunique()) if "keyword" in display_df.columns else 0,
    )

    columns = [column for column in NEWS_DISPLAY_COLUMNS if column in display_df.columns]
    if not columns:
        st.info("No news columns available for display.")
        return

    st.dataframe(
        display_df[columns].head(30),
        use_container_width=True,
        column_config={
            "open_link": column_config.LinkColumn("Open Article", display_text="기사 열기")
        },
        hide_index=True,
    )