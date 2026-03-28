from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime
from typing import Any

from src.data.database import SessionLocal
from src.data.db_repository import (
    get_latest_market_snapshots,
    get_latest_news_articles,
    get_latest_risk_snapshots,
    save_candle_snapshots,
    save_market_snapshots,
    save_news_articles,
    save_risk_snapshots,
)
from src.pipelines.data_pipeline import CryptoRiskDataPipeline
from src.utils.config import load_config


logger = logging.getLogger("realtime_pipeline")


def configure_logging() -> None:
    """
    Configure console logging for long-running realtime execution.
    """
    if logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RealtimeCryptoRiskPipeline:
    """
    Operational realtime pipeline for continuous risk ingestion and storage.

    Responsibilities
    ----------------
    1. Run the existing CryptoRiskDataPipeline.
    2. Persist market, news, and risk outputs into SQLite.
    3. Expose monitoring helpers for current DB state.
    4. Support continuous polling execution with graceful shutdown.

    Notes
    -----
    - This is a practical MVP near-real-time pipeline.
    - The current implementation uses polling rather than Kafka / Celery.
    - It is intentionally structured for future upgrades such as PostgreSQL,
      scheduler integration, and alert dispatch hooks.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.pipeline = CryptoRiskDataPipeline(config_path=config_path)
        self.is_running = True
        self.default_poll_interval_seconds = self._resolve_poll_interval_seconds()

    def _resolve_poll_interval_seconds(self) -> int:
        """
        Resolve polling interval from config.

        Current policy:
        - use alert evaluation minutes when present
        - otherwise default to 60 seconds
        """
        risk_config = self.config.get("risk", {})
        evaluation_minutes = risk_config.get("alert_evaluation_minutes")

        try:
            if evaluation_minutes is not None:
                resolved = int(float(evaluation_minutes) * 60)
                return max(resolved, 10)
        except (TypeError, ValueError):
            pass

        return 60

    def request_shutdown(self) -> None:
        """
        Request graceful shutdown of the polling loop.
        """
        logger.info("Shutdown requested. Realtime pipeline will stop after the current cycle.")
        self.is_running = False

    def run_once(self) -> dict[str, Any]:
        """
        Run one full collection → evaluation → DB persistence cycle.
        """
        cycle_started_at = datetime.utcnow()
        logger.info("Realtime cycle started.")

        result = self.pipeline.run()

        market_snapshot = result.get("market_snapshot", {})
        candles_df = market_snapshot.get("candles")
        news_df = result.get("news_df")
        risk_evaluation_df = result.get("risk_evaluation_df")

        db = SessionLocal()
        try:
            save_market_snapshots(
                db=db,
                ticker_df=market_snapshot.get("ticker"),
                candles_df=candles_df,
                orderbook_df=market_snapshot.get("orderbook"),
            )
            save_candle_snapshots(
                db=db,
                candles_df=candles_df,
                interval_type="minute1",
            )
            save_news_articles(db=db, news_df=news_df)
            save_risk_snapshots(db=db, risk_df=risk_evaluation_df)
        finally:
            db.close()

        risk_row_count = 0 if risk_evaluation_df is None else len(risk_evaluation_df)
        candle_row_count = 0 if candles_df is None else len(candles_df)
        logger.info(
            "Realtime cycle completed successfully. started_at=%s risk_rows=%s candle_rows=%s",
            cycle_started_at.isoformat(),
            risk_row_count,
            candle_row_count,
        )

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

    def run_forever(self, poll_interval_seconds: int | None = None) -> None:
        """
        Run the realtime pipeline continuously using polling.

        Parameters
        ----------
        poll_interval_seconds : int | None
            Number of seconds to wait between each cycle.
            When None, the value is resolved from config.
        """
        interval = self.default_poll_interval_seconds if poll_interval_seconds is None else int(poll_interval_seconds)
        if interval <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0.")

        logger.info("Starting realtime polling loop. poll_interval_seconds=%s", interval)

        cycle_count = 0
        while self.is_running:
            cycle_count += 1
            try:
                result = self.run_once()
                risk_df = result.get("risk_evaluation_df")
                record_count = 0 if risk_df is None else len(risk_df)

                summary = self.get_db_snapshot_summary()
                logger.info(
                    "Cycle %s committed to DB. risk_rows=%s db_summary=%s",
                    cycle_count,
                    record_count,
                    summary,
                )
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping realtime pipeline.")
                self.request_shutdown()
                break
            except Exception:
                logger.exception("Realtime cycle %s failed.", cycle_count)

            if not self.is_running:
                break

            time.sleep(interval)

        logger.info("Realtime polling loop stopped.")


_ACTIVE_PIPELINE: RealtimeCryptoRiskPipeline | None = None



def _handle_shutdown_signal(signum: int, frame: Any) -> None:
    """
    Handle SIGINT / SIGTERM for graceful shutdown.
    """
    del frame
    logger.info("Received shutdown signal: %s", signum)
    if _ACTIVE_PIPELINE is not None:
        _ACTIVE_PIPELINE.request_shutdown()
    else:
        sys.exit(0)



def run_realtime_pipeline(
    config_path: str = "configs/config.yaml",
    poll_interval_seconds: int | None = None,
    run_once_only: bool = False,
) -> dict[str, Any] | None:
    """
    Functional wrapper for running the realtime pipeline.
    """
    global _ACTIVE_PIPELINE

    configure_logging()
    pipeline = RealtimeCryptoRiskPipeline(config_path=config_path)
    _ACTIVE_PIPELINE = pipeline

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    if run_once_only:
        return pipeline.run_once()

    pipeline.run_forever(poll_interval_seconds=poll_interval_seconds)
    return None


if __name__ == "__main__":
    run_realtime_pipeline(run_once_only=False, poll_interval_seconds=None)