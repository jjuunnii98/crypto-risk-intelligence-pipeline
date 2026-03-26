from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.config import load_config


RISK_COMPONENT_KEYS = [
    "volatility_risk",
    "liquidity_risk",
    "sentiment_risk",
    "event_risk",
]

RISK_LEVEL_ORDER = ["normal", "caution", "warning", "critical"]
WARNING_LIKE_LEVELS = {"warning", "critical"}
CAUTION_LIKE_LEVELS = {"caution", "warning", "critical"}


@dataclass(frozen=True)
class ThresholdLevel:
    caution: float
    warning: float
    critical: float


@dataclass(frozen=True)
class RiskThresholds:
    volatility_risk: ThresholdLevel
    liquidity_risk: ThresholdLevel
    sentiment_risk: ThresholdLevel
    event_risk: ThresholdLevel
    total_risk: ThresholdLevel


@dataclass(frozen=True)
class AlertEvaluationResult:
    asset: str
    total_risk_score: float
    risk_level: str
    should_alert: bool
    triggered_components: list[str]
    trigger_reasons: list[str]
    component_scores: dict[str, float]
    component_levels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "should_alert": self.should_alert,
            "triggered_components": self.triggered_components,
            "trigger_reasons": self.trigger_reasons,
            "component_scores": self.component_scores,
            "component_levels": self.component_levels,
        }


class AlertRuleEngine:
    """
    Evaluate risk component scores against configured thresholds.

    Expected input example:
    {
        "asset": "KRW-BTC",
        "volatility_risk": 0.81,
        "liquidity_risk": 0.42,
        "sentiment_risk": 0.76,
        "event_risk": 0.69,
        "total_risk_score": 0.74,
    }
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        config = load_config(config_path)
        self.thresholds = self._load_thresholds(config)

    @staticmethod
    def _build_threshold_level(raw: dict[str, Any], field_name: str) -> ThresholdLevel:
        if not isinstance(raw, dict):
            raise ValueError(f"Threshold config for '{field_name}' must be a dictionary.")

        try:
            caution = float(raw["caution"])
            warning = float(raw["warning"])
            critical = float(raw["critical"])
        except KeyError as exc:
            raise ValueError(
                f"Threshold config for '{field_name}' must contain caution, warning, critical."
            ) from exc

        if not (0 <= caution <= warning <= critical <= 1.0):
            raise ValueError(
                f"Threshold values for '{field_name}' must satisfy 0 <= caution <= warning <= critical <= 1.0."
            )

        return ThresholdLevel(
            caution=caution,
            warning=warning,
            critical=critical,
        )

    def _load_thresholds(self, config: dict[str, Any]) -> RiskThresholds:
        raw_thresholds = config.get("risk", {}).get("thresholds", {})
        if not isinstance(raw_thresholds, dict):
            raise ValueError("risk.thresholds must be a dictionary in config.yaml")

        return RiskThresholds(
            volatility_risk=self._build_threshold_level(
                raw_thresholds.get("volatility_risk", {}),
                "volatility_risk",
            ),
            liquidity_risk=self._build_threshold_level(
                raw_thresholds.get("liquidity_risk", {}),
                "liquidity_risk",
            ),
            sentiment_risk=self._build_threshold_level(
                raw_thresholds.get("sentiment_risk", {}),
                "sentiment_risk",
            ),
            event_risk=self._build_threshold_level(
                raw_thresholds.get("event_risk", {}),
                "event_risk",
            ),
            total_risk=self._build_threshold_level(
                raw_thresholds.get("total_risk", {}),
                "total_risk",
            ),
        )

    @staticmethod
    def _normalize_score(value: Any, field_name: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score for '{field_name}' must be numeric.") from exc

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score for '{field_name}' must be between 0.0 and 1.0.")

        return round(score, 4)

    @staticmethod
    def _resolve_level(score: float, threshold: ThresholdLevel) -> str:
        if score >= threshold.critical:
            return "critical"
        if score >= threshold.warning:
            return "warning"
        if score >= threshold.caution:
            return "caution"
        return "normal"

    def _get_component_threshold(self, component_name: str) -> ThresholdLevel:
        try:
            return getattr(self.thresholds, component_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported risk component: {component_name}") from exc

    def _evaluate_component_levels(
        self,
        component_scores: dict[str, float],
    ) -> tuple[dict[str, str], list[str], list[str]]:
        component_levels: dict[str, str] = {}
        triggered_components: list[str] = []
        trigger_reasons: list[str] = []

        for component_name, score in component_scores.items():
            threshold = self._get_component_threshold(component_name)
            level = self._resolve_level(score, threshold)
            component_levels[component_name] = level

            if level != "normal":
                triggered_components.append(component_name)
                trigger_reasons.append(f"{component_name}={score:.2f} ({level})")

        return component_levels, triggered_components, trigger_reasons

    def _resolve_combined_level(
        self,
        total_level: str,
        component_levels: dict[str, str],
    ) -> str:
        """
        Resolve the final risk level using both total risk and component risk.

        Policy
        ------
        - total_risk_score is the primary signal.
        - component risk can escalate the final level, but only with moderation.
        - a single critical component does not automatically force final critical unless
          the total level is already elevated.
        """
        levels = list(component_levels.values())

        warning_like_count = sum(level in WARNING_LIKE_LEVELS for level in levels)
        caution_like_count = sum(level in CAUTION_LIKE_LEVELS for level in levels)
        has_critical_component = any(level == "critical" for level in levels)
        has_warning_component = any(level in WARNING_LIKE_LEVELS for level in levels)
        has_caution_component = any(level in CAUTION_LIKE_LEVELS for level in levels)

        if total_level == "critical":
            return "critical"

        if total_level == "warning":
            if has_critical_component or warning_like_count >= 2:
                return "critical"
            return "warning"

        if total_level == "caution":
            if has_critical_component or warning_like_count >= 2:
                return "warning"
            return "caution"

        # total_level == "normal"
        if warning_like_count >= 2:
            return "warning"
        if has_warning_component or caution_like_count >= 2:
            return "caution"
        if has_caution_component:
            return "caution"
        return "normal"

    def evaluate(self, risk_payload: dict[str, Any]) -> AlertEvaluationResult:
        if not isinstance(risk_payload, dict):
            raise ValueError("risk_payload must be a dictionary.")

        asset = str(risk_payload.get("asset", "UNKNOWN"))

        component_scores = {
            component_name: self._normalize_score(
                risk_payload.get(component_name, 0.0),
                component_name,
            )
            for component_name in RISK_COMPONENT_KEYS
        }

        total_risk_score = self._normalize_score(
            risk_payload.get("total_risk_score", 0.0),
            "total_risk_score",
        )

        component_levels, triggered_components, trigger_reasons = self._evaluate_component_levels(
            component_scores
        )

        total_level = self._resolve_level(total_risk_score, self.thresholds.total_risk)
        combined_level = self._resolve_combined_level(total_level, component_levels)

        if total_level != "normal":
            trigger_reasons.insert(0, f"total_risk_score={total_risk_score:.2f} ({total_level})")

        should_alert = combined_level in WARNING_LIKE_LEVELS

        return AlertEvaluationResult(
            asset=asset,
            total_risk_score=total_risk_score,
            risk_level=combined_level,
            should_alert=should_alert,
            triggered_components=triggered_components,
            trigger_reasons=trigger_reasons,
            component_scores=component_scores,
            component_levels=component_levels,
        )


def evaluate_alert_rules(
    risk_payload: dict[str, Any],
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """
    Convenience wrapper used by services, pipelines, or API layers.
    """
    engine = AlertRuleEngine(config_path=config_path)
    return engine.evaluate(risk_payload).to_dict()