

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.collectors.google_news_collector import GoogleNewsCollector
from src.collectors.naver_news_collector import NaverNewsCollector
from src.collectors.upbit_collector import UpbitCollector
from src.services.risk_service import evaluate_risk_payload
from src.utils.config import ensure_data_directories, load_config


NEWS_NEGATIVE_KEYWORDS = [
    "급락",
    "하락",
    "폭락",
    "규제",
    "해킹",
    "소송",
    "위기",
    "리스크",
    "불안",
    "악재",
    "중단",
    "파산",
    "조사",
    "충돌",
    "급감",
]

NEWS_EVENT_KEYWORDS = [
    "규제",
    "해킹",
    "상장폐지",
    "소송",
    "보안",
    "제재",
    "공시",
    "긴급",
    "파산",
    "etf",
    "금리",
    "전쟁",
]

NEWS_SEVERE_EVENT_KEYWORDS = [
    "해킹",
    "상장폐지",
    "파산",
    "긴급",
    "제재",
    "소송",
    "전쟁",
]

ASSET_KEYWORD_MAP = {
    "KRW-BTC": ["비트코인", "btc"],
    "KRW-ETH": ["이더리움", "eth"],
    "KRW-SOL": ["솔라나", "sol"],
}


@dataclass(frozen=True)
class DataPipelinePaths:
    raw_dir: Path
    processed_dir: Path
    events_dir: Path


class CryptoRiskDataPipeline:
    """
    End-to-end data pipeline for the Crypto Risk Intelligence MVP.

    Responsibilities
    ----------------
    1. Collect market data from Upbit.
    2. Collect text data from Google News and Naver News.
    3. Build lightweight risk features for each asset.
    4. Evaluate risk levels via RiskService.
    5. Save raw / processed / event outputs to local data directories.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        ensure_data_directories(self.config)

        self.paths = DataPipelinePaths(
            raw_dir=Path(self.config["paths"]["raw_dir"]),
            processed_dir=Path(self.config["paths"]["processed_dir"]),
            events_dir=Path(self.config["paths"]["events_dir"]),
        )

        self.target_markets: list[str] = list(self.config["assets"]["target_markets"])
        self.rolling_window_minutes: int = int(
            self.config["risk"]["rolling_window_minutes"]
        )

        self.upbit_collector = UpbitCollector(config_path=config_path)
        self.google_news_collector = GoogleNewsCollector(config_path=config_path)
        self.naver_news_collector = NaverNewsCollector(config_path=config_path)

    @staticmethod
    def _current_timestamp_slug() -> str:
        return pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def _clip_score(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
        lowered_text = str(text).lower()
        return any(keyword.lower() in lowered_text for keyword in keywords)

    @staticmethod
    def _ensure_utc_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
        """
        Normalize timestamps to UTC-aware pandas Timestamp.

        Rules:
        - tz-naive timestamps are assumed to be Asia/Seoul and converted to UTC.
        - tz-aware timestamps are converted to UTC.
        """
        if pd.isna(value):
            return pd.NaT

        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("Asia/Seoul").tz_convert("UTC")
        return ts.tz_convert("UTC")

    def _filter_recent_news(
        self,
        news_df: pd.DataFrame,
        latest_timestamp: Any,
    ) -> pd.DataFrame:
        """
        Filter news to the configured rolling risk window.
        """
        if news_df.empty or "published_at" not in news_df.columns:
            return news_df.copy()

        latest_ts_utc = self._ensure_utc_timestamp(latest_timestamp)
        if pd.isna(latest_ts_utc):
            return news_df.copy()

        window_start = latest_ts_utc - pd.Timedelta(minutes=self.rolling_window_minutes)
        filtered_df = news_df.loc[
            (news_df["published_at"].notna())
            & (news_df["published_at"] >= window_start)
            & (news_df["published_at"] <= latest_ts_utc)
        ].copy()
        return filtered_df

    def _extract_orderbook_spread(self, asset_orderbook: pd.DataFrame) -> float:
        """
        Extract best ask/bid spread ratio from orderbook_units when available.
        """
        if asset_orderbook.empty or "orderbook_units" not in asset_orderbook.columns:
            return 0.0

        orderbook_units = asset_orderbook.iloc[0].get("orderbook_units", [])
        if not isinstance(orderbook_units, list) or not orderbook_units:
            return 0.0

        best_unit = orderbook_units[0]
        ask_price = float(best_unit.get("ask_price", 0.0))
        bid_price = float(best_unit.get("bid_price", 0.0))

        if ask_price <= 0 or bid_price <= 0:
            return 0.0

        mid_price = (ask_price + bid_price) / 2.0
        if mid_price <= 0:
            return 0.0

        return max((ask_price - bid_price) / mid_price, 0.0)

    def collect_market_data(self) -> dict[str, pd.DataFrame]:
        """
        Collect ticker, orderbook, and minute candles from Upbit.
        """
        return self.upbit_collector.fetch_market_snapshot()

    def collect_news_data(self) -> pd.DataFrame:
        """
        Collect and merge Google News + Naver News.
        """
        frames: list[pd.DataFrame] = []

        google_df = self.google_news_collector.fetch_all()
        if not google_df.empty:
            frames.append(google_df)

        try:
            naver_df = self.naver_news_collector.fetch_all()
            if not naver_df.empty:
                frames.append(naver_df)
        except Exception:
            # Keep pipeline alive even if Naver credentials are missing or request fails.
            naver_df = pd.DataFrame()

        if not frames:
            return pd.DataFrame(
                columns=[
                    "keyword",
                    "title",
                    "link",
                    "published_at",
                    "collected_at",
                    "source",
                ]
            )

        news_df = pd.concat(frames, ignore_index=True)
        news_df = news_df.drop_duplicates(subset=["title", "link"]).reset_index(drop=True)

        if "published_at" in news_df.columns:
            news_df["published_at"] = pd.to_datetime(
                news_df["published_at"],
                errors="coerce",
                utc=True,
            )

        if "collected_at" in news_df.columns:
            news_df["collected_at"] = pd.to_datetime(
                news_df["collected_at"],
                errors="coerce",
                utc=True,
            )

        news_df = news_df.sort_values("published_at", ascending=False).reset_index(drop=True)
        return news_df

    def _save_dataframe(self, df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def save_raw_data(
        self,
        market_snapshot: dict[str, pd.DataFrame],
        news_df: pd.DataFrame,
        timestamp_slug: str,
    ) -> dict[str, str]:
        """
        Persist raw collector outputs for traceability.
        """
        saved_paths: dict[str, str] = {}

        for name, df in market_snapshot.items():
            if not df.empty:
                path = self._save_dataframe(
                    df,
                    self.paths.raw_dir,
                    f"{name}_{timestamp_slug}.csv",
                )
                saved_paths[name] = str(path)

        if not news_df.empty:
            news_path = self._save_dataframe(
                news_df,
                self.paths.raw_dir,
                f"news_{timestamp_slug}.csv",
            )
            saved_paths["news"] = str(news_path)

        return saved_paths

    def _filter_asset_news(self, news_df: pd.DataFrame, market: str) -> pd.DataFrame:
        """
        Filter news rows relevant to a given market using keyword matching.
        """
        if news_df.empty:
            return news_df.copy()

        asset_keywords = ASSET_KEYWORD_MAP.get(market, [])
        if not asset_keywords:
            return pd.DataFrame(columns=news_df.columns)

        title_mask = news_df["title"].fillna("").apply(
            lambda title: self._contains_any_keyword(title, asset_keywords)
        )

        keyword_mask = pd.Series(False, index=news_df.index)
        if "keyword" in news_df.columns:
            keyword_mask = news_df["keyword"].fillna("").apply(
                lambda keyword: self._contains_any_keyword(keyword, asset_keywords)
            )

        return news_df.loc[title_mask | keyword_mask].copy()

    def _compute_volatility_risk(self, asset_candles: pd.DataFrame) -> float:
        if asset_candles.empty or len(asset_candles) < 20:
            return 0.0

        recent_candles = asset_candles.tail(self.rolling_window_minutes).copy()
        returns = recent_candles["trade_price"].pct_change().dropna()
        if returns.empty:
            return 0.0

        volatility_1h = float(returns.std())
        scaled_score = volatility_1h / 0.015

        latest_row = recent_candles.iloc[-1]
        rsi_penalty = 0.0
        if "rsi_14" in recent_candles.columns and pd.notna(latest_row.get("rsi_14")):
            latest_rsi = float(latest_row["rsi_14"])
            if latest_rsi >= 75 or latest_rsi <= 25:
                rsi_penalty = 0.10

        bollinger_penalty = 0.0
        if (
            "bollinger_bandwidth_20" in recent_candles.columns
            and pd.notna(latest_row.get("bollinger_bandwidth_20"))
        ):
            latest_bandwidth = float(latest_row["bollinger_bandwidth_20"])
            bollinger_penalty = min(latest_bandwidth * 1.5, 0.20)

        volatility_risk = scaled_score + rsi_penalty + bollinger_penalty
        return self._clip_score(volatility_risk)

    def _compute_liquidity_risk(self, orderbook_df: pd.DataFrame, market: str) -> float:
        if orderbook_df.empty:
            return 0.0

        asset_orderbook = orderbook_df.loc[orderbook_df["market"] == market].copy()
        if asset_orderbook.empty:
            return 0.0

        row = asset_orderbook.iloc[0]
        total_ask_size = float(row.get("total_ask_size", 0.0))
        total_bid_size = float(row.get("total_bid_size", 0.0))
        total_depth = total_ask_size + total_bid_size

        imbalance = abs(total_bid_size - total_ask_size) / total_depth if total_depth > 0 else 0.0
        shallow_book_penalty = 1.0 - min(total_depth / 20000.0, 1.0)
        spread_ratio = self._extract_orderbook_spread(asset_orderbook)
        spread_penalty = min(spread_ratio / 0.002, 1.0)

        liquidity_risk = 0.5 * imbalance + 0.25 * shallow_book_penalty + 0.25 * spread_penalty
        return self._clip_score(liquidity_risk)

    def _compute_sentiment_risk(self, asset_news_df: pd.DataFrame) -> float:
        if asset_news_df.empty:
            return 0.0

        titles = asset_news_df["title"].fillna("")
        negative_count = titles.apply(
            lambda title: self._contains_any_keyword(title, NEWS_NEGATIVE_KEYWORDS)
        ).sum()

        negative_ratio = self._safe_divide(float(negative_count), float(len(asset_news_df)))
        volume_penalty = min(float(len(asset_news_df)) / 20.0, 0.20)
        sentiment_risk = negative_ratio + volume_penalty
        return self._clip_score(sentiment_risk)

    def _compute_event_risk(self, asset_news_df: pd.DataFrame) -> float:
        if asset_news_df.empty:
            return 0.0

        titles = asset_news_df["title"].fillna("")
        event_count = titles.apply(
            lambda title: self._contains_any_keyword(title, NEWS_EVENT_KEYWORDS)
        ).sum()
        severe_event_count = titles.apply(
            lambda title: self._contains_any_keyword(title, NEWS_SEVERE_EVENT_KEYWORDS)
        ).sum()

        base_event_score = min(float(event_count) / 10.0, 1.0)
        severe_bonus = min(float(severe_event_count) * 0.15, 0.45)
        event_risk = base_event_score + severe_bonus
        return self._clip_score(event_risk)

    def build_asset_risk_features(
        self,
        candles_df: pd.DataFrame,
        orderbook_df: pd.DataFrame,
        news_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build one-row risk feature summary per target market.
        """
        if candles_df.empty:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []

        for market in self.target_markets:
            asset_candles = candles_df.loc[candles_df["market"] == market].copy()
            asset_candles = asset_candles.sort_values("timestamp_kst").reset_index(drop=True)

            asset_news = self._filter_asset_news(news_df, market)

            latest_timestamp = (
                asset_candles["timestamp_kst"].max()
                if not asset_candles.empty and "timestamp_kst" in asset_candles.columns
                else pd.NaT
            )
            recent_asset_news = self._filter_recent_news(asset_news, latest_timestamp)

            volatility_risk = self._compute_volatility_risk(asset_candles)
            liquidity_risk = self._compute_liquidity_risk(orderbook_df, market)
            sentiment_risk = self._compute_sentiment_risk(recent_asset_news)
            event_risk = self._compute_event_risk(recent_asset_news)

            records.append(
                {
                    "asset": market,
                    "latest_timestamp": latest_timestamp,
                    "volatility_risk": round(volatility_risk, 4),
                    "liquidity_risk": round(liquidity_risk, 4),
                    "sentiment_risk": round(sentiment_risk, 4),
                    "event_risk": round(event_risk, 4),
                    "news_count": int(len(recent_asset_news)),
                    "negative_news_count": int(
                        recent_asset_news["title"].fillna("").apply(
                            lambda title: self._contains_any_keyword(title, NEWS_NEGATIVE_KEYWORDS)
                        ).sum()
                    ) if not recent_asset_news.empty else 0,
                    "event_news_count": int(
                        recent_asset_news["title"].fillna("").apply(
                            lambda title: self._contains_any_keyword(title, NEWS_EVENT_KEYWORDS)
                        ).sum()
                    ) if not recent_asset_news.empty else 0,
                }
            )

        return pd.DataFrame(records)

    def evaluate_risk(self, risk_feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate per-asset risk scores through RiskService.
        """
        if risk_feature_df.empty:
            return pd.DataFrame()

        results: list[dict[str, Any]] = []

        for record in risk_feature_df.to_dict(orient="records"):
            evaluation = evaluate_risk_payload(record, config_path=self.config_path)
            results.append(
                {
                    **record,
                    **evaluation,
                }
            )

        return pd.DataFrame(results)

    def save_processed_outputs(
        self,
        risk_feature_df: pd.DataFrame,
        risk_evaluation_df: pd.DataFrame,
        timestamp_slug: str,
    ) -> dict[str, str]:
        saved_paths: dict[str, str] = {}

        if not risk_feature_df.empty:
            feature_path = self._save_dataframe(
                risk_feature_df,
                self.paths.processed_dir,
                f"risk_features_{timestamp_slug}.csv",
            )
            saved_paths["risk_features"] = str(feature_path)

        if not risk_evaluation_df.empty:
            evaluation_path = self._save_dataframe(
                risk_evaluation_df,
                self.paths.events_dir,
                f"risk_events_{timestamp_slug}.csv",
            )
            saved_paths["risk_events"] = str(evaluation_path)

        return saved_paths

    def run(self) -> dict[str, Any]:
        """
        Run the end-to-end data pipeline.
        """
        timestamp_slug = self._current_timestamp_slug()

        market_snapshot = self.collect_market_data()
        news_df = self.collect_news_data()

        raw_paths = self.save_raw_data(
            market_snapshot=market_snapshot,
            news_df=news_df,
            timestamp_slug=timestamp_slug,
        )

        candles_df = market_snapshot.get("candles", pd.DataFrame())
        orderbook_df = market_snapshot.get("orderbook", pd.DataFrame())

        risk_feature_df = self.build_asset_risk_features(
            candles_df=candles_df,
            orderbook_df=orderbook_df,
            news_df=news_df,
        )
        risk_evaluation_df = self.evaluate_risk(risk_feature_df)

        processed_paths = self.save_processed_outputs(
            risk_feature_df=risk_feature_df,
            risk_evaluation_df=risk_evaluation_df,
            timestamp_slug=timestamp_slug,
        )

        return {
            "timestamp_slug": timestamp_slug,
            "raw_paths": raw_paths,
            "processed_paths": processed_paths,
            "market_snapshot": market_snapshot,
            "news_df": news_df,
            "risk_feature_df": risk_feature_df,
            "risk_evaluation_df": risk_evaluation_df,
        }


def run_data_pipeline(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    pipeline = CryptoRiskDataPipeline(config_path=config_path)
    return pipeline.run()


if __name__ == "__main__":
    result = run_data_pipeline()

    print("\n[DATA PIPELINE COMPLETED]")
    print("Raw files:", result["raw_paths"])
    print("Processed files:", result["processed_paths"])

    risk_df = result["risk_evaluation_df"]
    if not risk_df.empty:
        print("\n[RISK EVALUATION SAMPLE]")
        print(
            risk_df[[
                "asset",
                "total_risk_score",
                "risk_level",
                "should_alert",
                "trigger_reasons",
            ]]
        )
    else:
        print("\nNo risk evaluation output generated.")