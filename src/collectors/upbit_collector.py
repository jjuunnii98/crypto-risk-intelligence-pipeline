from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from src.utils.config import (
    get_target_markets,
    get_upbit_base_url,
    load_config,
)


@dataclass
class UpbitCollectorConfig:
    base_url: str
    markets: list[str]
    interval_minutes: int
    lookback_candles: int
    timeout_seconds: int
    pause_seconds: float


# Technical indicator defaults
DEFAULT_SHORT_MA_WINDOW = 20
DEFAULT_LONG_MA_WINDOW = 60
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_WINDOW = 20
DEFAULT_BOLLINGER_STD = 2.0


class UpbitCollector:
    """
    Collector for Upbit market data.

    Supports:
    - ticker
    - minute candles
    - orderbook
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        config = load_config(config_path)

        self._config = UpbitCollectorConfig(
            base_url=get_upbit_base_url(config),
            markets=get_target_markets(config),
            interval_minutes=int(config["collection"]["market"]["interval_minutes"]),
            lookback_candles=int(config["collection"]["market"]["lookback_candles"]),
            timeout_seconds=int(config["collection"]["market"]["request_timeout_seconds"]),
            pause_seconds=float(config["collection"]["market"]["pause_seconds_between_requests"]),
        )

    @property
    def markets(self) -> list[str]:
        return self._config.markets

    @staticmethod
    def _format_price_with_commas(value: Any) -> str:
        """
        Format numeric price values with comma separators.
        """
        if pd.isna(value):
            return ""

        numeric_value = float(value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):,}"
        return f"{numeric_value:,.2f}"

    @staticmethod
    def _format_numeric_with_precision(value: Any, decimals: int = 2) -> str:
        """
        Format general numeric values with comma separators and fixed precision.
        """
        if pd.isna(value):
            return ""
        return f"{float(value):,.{decimals}f}"

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = DEFAULT_RSI_WINDOW) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis features to candle dataframe.

        Features added:
        - short_ma_20
        - long_ma_60
        - rsi_14
        - bollinger_mid_20
        - bollinger_upper_20
        - bollinger_lower_20
        - bollinger_bandwidth_20
        """
        if df.empty:
            return df

        enriched_df = df.copy()

        price_series = enriched_df["trade_price"]

        enriched_df["ma_20"] = price_series.rolling(
            window=DEFAULT_SHORT_MA_WINDOW,
            min_periods=DEFAULT_SHORT_MA_WINDOW,
        ).mean()
        enriched_df["ma_60"] = price_series.rolling(
            window=DEFAULT_LONG_MA_WINDOW,
            min_periods=DEFAULT_LONG_MA_WINDOW,
        ).mean()

        enriched_df["rsi_14"] = self._compute_rsi(
            price_series,
            window=DEFAULT_RSI_WINDOW,
        )

        rolling_mean = price_series.rolling(
            window=DEFAULT_BOLLINGER_WINDOW,
            min_periods=DEFAULT_BOLLINGER_WINDOW,
        ).mean()
        rolling_std = price_series.rolling(
            window=DEFAULT_BOLLINGER_WINDOW,
            min_periods=DEFAULT_BOLLINGER_WINDOW,
        ).std()

        enriched_df["bollinger_mid_20"] = rolling_mean
        enriched_df["bollinger_upper_20"] = rolling_mean + (rolling_std * DEFAULT_BOLLINGER_STD)
        enriched_df["bollinger_lower_20"] = rolling_mean - (rolling_std * DEFAULT_BOLLINGER_STD)
        enriched_df["bollinger_bandwidth_20"] = (
            enriched_df["bollinger_upper_20"] - enriched_df["bollinger_lower_20"]
        ) / enriched_df["bollinger_mid_20"].replace(0, pd.NA)

        return enriched_df

    def _add_formatted_technical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable formatted columns for technical indicators.
        """
        if df.empty:
            return df

        enriched_df = df.copy()

        price_like_columns = [
            "trade_price",
            "ma_20",
            "ma_60",
            "bollinger_mid_20",
            "bollinger_upper_20",
            "bollinger_lower_20",
        ]
        for column in price_like_columns:
            if column in enriched_df.columns:
                enriched_df[f"{column}_formatted"] = enriched_df[column].apply(
                    self._format_price_with_commas
                )

        if "rsi_14" in enriched_df.columns:
            enriched_df["rsi_14_formatted"] = enriched_df["rsi_14"].apply(
                lambda x: self._format_numeric_with_precision(x, decimals=2)
            )

        if "bollinger_bandwidth_20" in enriched_df.columns:
            enriched_df["bollinger_bandwidth_20_formatted"] = enriched_df[
                "bollinger_bandwidth_20"
            ].apply(lambda x: self._format_numeric_with_precision(x, decimals=4))

        return enriched_df

    def _request(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Internal GET request helper for Upbit REST API.
        """
        url = f"{self._config.base_url}/{endpoint}"

        response = requests.get(
            url,
            params=params,
            timeout=self._config.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected response format from Upbit API: {type(payload)}")

        return payload

    def fetch_ticker(self, markets: list[str] | None = None) -> pd.DataFrame:
        """
        Fetch current ticker snapshot for one or more markets.
        """
        target_markets = markets or self._config.markets

        payload = self._request(
            endpoint="ticker",
            params={"markets": ",".join(target_markets)},
        )

        df = pd.DataFrame(payload)
        if df.empty:
            return df

        df["collected_at"] = pd.Timestamp.now(tz="Asia/Seoul")
        if "trade_price" in df.columns:
            df["trade_price_formatted"] = df["trade_price"].apply(self._format_price_with_commas)
        return df

    def fetch_orderbook(self, markets: list[str] | None = None) -> pd.DataFrame:
        """
        Fetch current orderbook snapshot.
        """
        target_markets = markets or self._config.markets

        payload = self._request(
            endpoint="orderbook",
            params={"markets": ",".join(target_markets)},
        )

        df = pd.DataFrame(payload)
        if df.empty:
            return df

        df["collected_at"] = pd.Timestamp.now(tz="Asia/Seoul")
        if "total_ask_size" in df.columns:
            df["total_ask_size_formatted"] = df["total_ask_size"].apply(
                lambda x: self._format_numeric_with_precision(x, decimals=6)
            )
        if "total_bid_size" in df.columns:
            df["total_bid_size_formatted"] = df["total_bid_size"].apply(
                lambda x: self._format_numeric_with_precision(x, decimals=6)
            )
        return df

    def fetch_minute_candles(
        self,
        market: str,
        unit: int | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch minute candles for a single market.

        Parameters
        ----------
        market : str
            Market code such as KRW-BTC
        unit : int | None
            Candle unit in minutes (default: config interval)
        count : int | None
            Number of candles (default: config lookback)
        """
        candle_unit = unit or self._config.interval_minutes
        candle_count = count or self._config.lookback_candles

        endpoint = f"candles/minutes/{candle_unit}"
        payload = self._request(
            endpoint=endpoint,
            params={
                "market": market,
                "count": candle_count,
            },
        )

        df = pd.DataFrame(payload)
        if df.empty:
            return df

        df["market"] = market
        df["collected_at"] = pd.Timestamp.now(tz="Asia/Seoul")

        if "candle_date_time_kst" in df.columns:
            df["timestamp_kst"] = pd.to_datetime(df["candle_date_time_kst"])
        elif "candle_date_time_utc" in df.columns:
            df["timestamp_kst"] = (
                pd.to_datetime(df["candle_date_time_utc"])
                .dt.tz_localize("UTC")
                .dt.tz_convert("Asia/Seoul")
            )
        else:
            raise ValueError("Upbit candle response does not contain a valid timestamp column.")

        df = df.sort_values("timestamp_kst").reset_index(drop=True)
        df = self._add_technical_indicators(df)
        df = self._add_formatted_technical_columns(df)

        return df

    def fetch_all_market_candles(self) -> pd.DataFrame:
        """
        Fetch minute candles for all configured target markets.
        """
        frames: list[pd.DataFrame] = []

        for market in self._config.markets:
            df = self.fetch_minute_candles(market=market)
            if not df.empty:
                frames.append(df)
            time.sleep(self._config.pause_seconds)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=0, ignore_index=True)

    def fetch_market_snapshot(self) -> dict[str, pd.DataFrame]:
        """
        Collect a full market snapshot:
        - ticker
        - orderbook
        - candles
        """
        snapshot = {
            "ticker": self.fetch_ticker(),
            "orderbook": self.fetch_orderbook(),
            "candles": self.fetch_all_market_candles(),
        }
        return snapshot


def main() -> None:
    """
    Simple manual test entrypoint.
    """
    collector = UpbitCollector()

    print("Target markets:", collector.markets)

    ticker_df = collector.fetch_ticker()
    print("\n[TICKER]")
    print(ticker_df[["market", "trade_price_formatted", "signed_change_rate"]].head())

    orderbook_df = collector.fetch_orderbook()
    print("\n[ORDERBOOK]")
    print(orderbook_df[["market", "total_ask_size_formatted", "total_bid_size_formatted"]].head())

    candles_df = collector.fetch_all_market_candles()
    print("\n[CANDLES + TECHNICAL INDICATORS]")
    print(
        candles_df[[
            "market",
            "timestamp_kst",
            "trade_price_formatted",
            "candle_acc_trade_volume",
            "ma_20_formatted",
            "ma_60_formatted",
            "rsi_14_formatted",
            "bollinger_upper_20_formatted",
            "bollinger_lower_20_formatted",
        ]].head(25)
    )


if __name__ == "__main__":
    main()