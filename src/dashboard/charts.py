from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


# =========================
# Candle Interval 설정
# =========================
CANDLE_INTERVAL_RULES: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "240m": "240min",
    "1D": "1D",
    "7D": "7D",
    "30D": "30D",
}

CANDLE_INTERVAL_OPTIONS: list[str] = list(CANDLE_INTERVAL_RULES.keys())


# =========================
# Risk Trend
# =========================
def render_total_risk_timeseries(risk_history_df: pd.DataFrame) -> None:
    st.subheader("Historical Total Risk Trend")

    if risk_history_df.empty:
        st.info("No historical risk snapshots available yet.")
        return

    df = risk_history_df.sort_values("created_at").tail(180).copy()

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Timestamp", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("total_risk_score:Q", title="Total Risk Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("asset:N", title="Asset"),
            tooltip=[
                alt.Tooltip("created_at:T", title="Timestamp"),
                alt.Tooltip("asset:N", title="Asset"),
                alt.Tooltip("total_risk_score:Q", title="Total Risk", format=".4f"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def render_risk_change_timeseries(risk_history_df: pd.DataFrame) -> None:
    st.subheader("Historical Risk Change Trend")

    if risk_history_df.empty or "risk_change" not in risk_history_df.columns:
        st.info("No risk change data.")
        return

    df = (
        risk_history_df.dropna(subset=["risk_change"])
        .sort_values("created_at")
        .tail(180)
        .copy()
    )

    if df.empty:
        st.info("Risk change becomes visible after multiple snapshots are accumulated.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Timestamp", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("risk_change:Q", title="Δ Total Risk Score"),
            color=alt.Color("asset:N", title="Asset"),
            tooltip=[
                alt.Tooltip("created_at:T", title="Timestamp"),
                alt.Tooltip("asset:N", title="Asset"),
                alt.Tooltip("risk_change:Q", title="Δ Risk", format=".4f"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


# =========================
# Market Price
# =========================
def render_market_price_timeseries(market_history_df: pd.DataFrame) -> None:
    st.subheader("Historical Asset Price Trend")

    if market_history_df.empty or "trade_price" not in market_history_df.columns:
        st.info("No price data.")
        return

    df = market_history_df.dropna(subset=["trade_price", "market", "collected_at"]).copy()
    if df.empty:
        st.info("No valid market price observations available.")
        return

    df = df.sort_values(["market", "collected_at"])
    df["base_price"] = df.groupby("market")["trade_price"].transform("first")
    df = df.loc[df["base_price"] > 0].copy()
    df["normalized"] = (df["trade_price"] / df["base_price"]) * 100
    df = df.tail(300)

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("collected_at:T", title="Timestamp", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("normalized:Q", title="Normalized Price Index (Base=100)"),
            color=alt.Color("market:N", title="Asset"),
            tooltip=[
                alt.Tooltip("collected_at:T", title="Timestamp"),
                alt.Tooltip("market:N", title="Asset"),
                alt.Tooltip("trade_price:Q", title="Trade Price", format=",.0f"),
                alt.Tooltip("normalized:Q", title="Index", format=".2f"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


# =========================
# Candle Resample
# =========================
def _resample_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df

    rule = CANDLE_INTERVAL_RULES.get(interval, "1min")

    working_df = df.copy()
    working_df = working_df.sort_values("candle_time_kst")
    working_df = working_df.dropna(subset=["candle_time_kst"])
    if working_df.empty:
        return working_df

    working_df = working_df.set_index("candle_time_kst")

    agg = {
        "opening_price": "first",
        "high_price": "max",
        "low_price": "min",
        "trade_price": "last",
    }

    if "candle_acc_trade_volume" in working_df.columns:
        agg["candle_acc_trade_volume"] = "sum"
    if "market" in working_df.columns:
        agg["market"] = "last"

    resampled_df = working_df.resample(rule).agg(agg)
    resampled_df = resampled_df.dropna(subset=["opening_price", "high_price", "low_price", "trade_price"])
    resampled_df = resampled_df.reset_index()

    if "market" not in resampled_df.columns and "market" in df.columns and not df.empty:
        resampled_df["market"] = df["market"].iloc[0]

    return resampled_df


# =========================
# Candlestick Chart
# =========================
def render_candlestick_chart(
    candle_df: pd.DataFrame,
    market: str,
    interval: str = "1m",
) -> None:
    st.subheader(f"Candlestick Chart ({market} / {interval})")

    if candle_df.empty:
        st.info("No candle data.")
        return

    df = _resample_candles(candle_df, interval)
    if df.empty:
        st.info("No resampled candle data.")
        return

    max_points = 120 if interval in {"1m", "3m", "5m", "10m", "15m"} else 90
    df = df.sort_values("candle_time_kst").tail(max_points).copy()

    df["is_bull"] = df["trade_price"] >= df["opening_price"]
    df["body_low"] = df[["opening_price", "trade_price"]].min(axis=1)
    df["body_high"] = df[["opening_price", "trade_price"]].max(axis=1)
    df["candle_color"] = df["is_bull"].map({True: "bull", False: "bear"})

    price_min = float(df["low_price"].min())
    price_max = float(df["high_price"].max())
    price_padding = max((price_max - price_min) * 0.08, max(price_max * 0.002, 1.0))
    price_scale = alt.Scale(domain=[price_min - price_padding, price_max + price_padding], zero=False)

    x_axis = alt.X(
        "candle_time_kst:T",
        title="Timestamp",
        axis=alt.Axis(labelOverlap=True, format="%m-%d %H:%M"),
    )

    wick = (
        alt.Chart(df)
        .mark_rule(strokeWidth=1.5)
        .encode(
            x=x_axis,
            y=alt.Y("low_price:Q", title="Price", scale=price_scale),
            y2="high_price:Q",
            color=alt.Color(
                "candle_color:N",
                scale=alt.Scale(domain=["bull", "bear"], range=["#2563eb", "#dc2626"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("candle_time_kst:T", title="Timestamp"),
                alt.Tooltip("opening_price:Q", title="Open", format=",.0f"),
                alt.Tooltip("high_price:Q", title="High", format=",.0f"),
                alt.Tooltip("low_price:Q", title="Low", format=",.0f"),
                alt.Tooltip("trade_price:Q", title="Close", format=",.0f"),
            ],
        )
    )

    body = (
        alt.Chart(df)
        .mark_bar(size=10)
        .encode(
            x=x_axis,
            y=alt.Y("body_low:Q", title="Price", scale=price_scale),
            y2="body_high:Q",
            color=alt.Color(
                "candle_color:N",
                scale=alt.Scale(domain=["bull", "bear"], range=["#2563eb", "#dc2626"]),
                legend=None,
            ),
        )
    )

    candle_chart = alt.layer(wick, body).properties(height=520)
    st.altair_chart(candle_chart, use_container_width=True)

    if "candle_acc_trade_volume" in df.columns:
        volume_df = df.dropna(subset=["candle_acc_trade_volume"]).copy()
        if not volume_df.empty:
            volume_chart = (
                alt.Chart(volume_df)
                .mark_bar(size=10)
                .encode(
                    x=x_axis,
                    y=alt.Y("candle_acc_trade_volume:Q", title="Volume"),
                    color=alt.Color(
                        "candle_color:N",
                        scale=alt.Scale(domain=["bull", "bear"], range=["#93c5fd", "#fca5a5"]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("candle_time_kst:T", title="Timestamp"),
                        alt.Tooltip("candle_acc_trade_volume:Q", title="Volume", format=",.4f"),
                    ],
                )
                .properties(height=180)
            )
            st.altair_chart(volume_chart, use_container_width=True)