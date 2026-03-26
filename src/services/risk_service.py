

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.alerts.alert_rules import evaluate_alert_rules


RISK_COMPONENT_KEYS = [
    "volatility_risk",
    "liquidity_risk",
    "sentiment_risk",
    "event_risk",
]

DEFAULT_COMPONENT_WEIGHTS = {
    "volatility_risk": 0.35,
    "liquidity_risk": 0.20,
    "sentiment_risk": 0.20,
    "event_risk": 0.25,
}


@dataclass(frozen=True)
class RiskServiceResult:
    asset: str
    total_risk_score: float
    risk_level: str
    should_alert: bool
    component_scores: dict[str, float]
    triggered_components: list[str]
    trigger_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "should_alert": self.should_alert,
            "component_scores": self.component_scores,
            "triggered_components": self.triggered_components,
            "trigger_reasons": self.trigger_reasons,
        }


class RiskService:
    """
    Service layer for assembling risk payloads and evaluating alert rules.

    This service does not depend on model training artifacts yet.
    It accepts already-computed component risk scores and:
    1. validates them,
    2. computes total_risk_score when missing,
    3. evaluates thresholds through AlertRuleEngine.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.component_weights = DEFAULT_COMPONENT_WEIGHTS.copy()

    @staticmethod
    def _normalize_score(value: Any, field_name: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric.") from exc

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0.")

        return round(score, 4)

    def _normalize_component_scores(self, payload: dict[str, Any]) -> dict[str, float]:
        return {
            component_name: self._normalize_score(
                payload.get(component_name, 0.0),
                component_name,
            )
            for component_name in RISK_COMPONENT_KEYS
        }

    def _compute_total_risk_score(self, component_scores: dict[str, float]) -> float:
        weighted_sum = sum(
            component_scores[component_name] * self.component_weights[component_name]
            for component_name in RISK_COMPONENT_KEYS
        )
        return round(weighted_sum, 4)

    def build_risk_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize component scores and ensure total_risk_score exists.

        Expected input example:
        {
            "asset": "KRW-BTC",
            "volatility_risk": 0.81,
            "liquidity_risk": 0.42,
            "sentiment_risk": 0.76,
            "event_risk": 0.69
        }
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary.")

        asset = str(payload.get("asset", "UNKNOWN"))
        component_scores = self._normalize_component_scores(payload)

        if "total_risk_score" in payload and payload.get("total_risk_score") is not None:
            total_risk_score = self._normalize_score(
                payload.get("total_risk_score"),
                "total_risk_score",
            )
        else:
            total_risk_score = self._compute_total_risk_score(component_scores)

        risk_payload = {
            "asset": asset,
            **component_scores,
            "total_risk_score": total_risk_score,
        }
        return risk_payload

    def evaluate(self, payload: dict[str, Any]) -> RiskServiceResult:
        """
        Build a validated risk payload and evaluate alert rules.
        """
        risk_payload = self.build_risk_payload(payload)
        evaluation = evaluate_alert_rules(
            risk_payload=risk_payload,
            config_path=self.config_path,
        )

        return RiskServiceResult(
            asset=evaluation["asset"],
            total_risk_score=evaluation["total_risk_score"],
            risk_level=evaluation["risk_level"],
            should_alert=evaluation["should_alert"],
            component_scores=evaluation["component_scores"],
            triggered_components=evaluation["triggered_components"],
            trigger_reasons=evaluation["trigger_reasons"],
        )

    def evaluate_to_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Convenience method for API, pipeline, and dashboard layers.
        """
        return self.evaluate(payload).to_dict()


def evaluate_risk_payload(
    payload: dict[str, Any],
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """
    Functional wrapper for quick integration in routes or pipelines.
    """
    service = RiskService(config_path=config_path)
    return service.evaluate_to_dict(payload)