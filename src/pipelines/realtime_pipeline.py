

from __future__ import annotations

import time
from typing import Any

from src.data.database import SessionLocal
from src.data.db_repository import (
    get_latest_market_snapshots,
    get_latest_news_articles,
    get_latest_risk_snapshots,
    save_market_snapshots,
    save_news_articles,
    save_risk_snapshots,
)
from src.pipelines.data_pipeline import CryptoRiskDataPipeline


class RealtimeCryptoRiskPipeline:
    """
    Realtime pipeline that bridges the CSV-based data pipeline to the SQLite database.

    Responsibilities
    ----------------
    1. Run the existing CryptoRiskDataPipeline.
    2. Persist market, news, and risk outputs into SQLite.
    3. Provide simple monitoring helpers for latest DB state.

    Notes
    -----
    - This is a practical MVP near-real-time pipeline.
    - The initial implementation uses polling instead of Kafka / Celery.
    - It is designed to be upgraded later to PostgreSQL and scheduled workers.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.pipeline = CryptoRiskDataPipeline(config_path=config_path)

    def run_once(self) -> dict[str, Any]:
        """
        Run one full collection → evaluation → DB persistence cycle.
        """
        result = self.pipeline.run()

        market_snapshot = result.get("market_snapshot", {})
        news_df = result.get("news_df")
        risk_evaluation_df = result.get("risk_evaluation_df")

        db = SessionLocal()
        try:
            save_market_snapshots(
                db=db,
                ticker_df=market_snapshot.get("ticker"),
                candles_df=market_snapshot.get("candles"),
                orderbook_df=market_snapshot.get("orderbook"),
            )
            save_news_articles(db=db, news_df=news_df)
            save_risk_snapshots(db=db, risk_df=risk_evaluation_df)
        finally:
            db.close()

        return result

    def get_db_snapshot_summary(self) -> dict[str, Any]:
        """
        Return a lightweight summary of the current DB state.
        """
        db = SessionLocal()
        try:
            market_rows = get_latest_market_snapshots(db)
            news_rows = get_latest_news_articles(db)
            risk_rows = get_latest_risk_snapshots(db)
        finally:
            db.close()

        return {
            "market_snapshot_rows": len(market_rows),
            "news_article_rows": len(news_rows),
            "risk_snapshot_rows": len(risk_rows),
        }

    def run_forever(self, poll_interval_seconds: int = 60) -> None:
        """
        Run the realtime pipeline continuously using polling.

        Parameters
        ----------
        poll_interval_seconds : int
            Number of seconds to wait between each cycle.
        """
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0.")

        print("[REALTIME PIPELINE] Starting polling loop...")
        print(f"[REALTIME PIPELINE] Poll interval: {poll_interval_seconds} seconds")

        while True:
            try:
                result = self.run_once()
                risk_df = result.get("risk_evaluation_df")
                record_count = 0 if risk_df is None else len(risk_df)

                print("[REALTIME PIPELINE] Cycle completed successfully.")
                print(f"[REALTIME PIPELINE] Risk rows written this cycle: {record_count}")

                summary = self.get_db_snapshot_summary()
                print(f"[REALTIME PIPELINE] DB summary: {summary}")
            except KeyboardInterrupt:
                print("[REALTIME PIPELINE] Interrupted by user. Stopping...")
                break
            except Exception as exc:
                print(f"[REALTIME PIPELINE] Cycle failed: {exc}")

            time.sleep(poll_interval_seconds)


def run_realtime_pipeline(
    config_path: str = "configs/config.yaml",
    poll_interval_seconds: int = 60,
    run_once_only: bool = False,
) -> dict[str, Any] | None:
    """
    Functional wrapper for running the realtime pipeline.
    """
    pipeline = RealtimeCryptoRiskPipeline(config_path=config_path)

    if run_once_only:
        return pipeline.run_once()

    pipeline.run_forever(poll_interval_seconds=poll_interval_seconds)
    return None


if __name__ == "__main__":
    run_realtime_pipeline(run_once_only=False, poll_interval_seconds=60)