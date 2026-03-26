

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ASSET_KEYWORD_MAP = {
    "KRW-BTC": ["비트코인", "btc"],
    "KRW-ETH": ["이더리움", "eth"],
    "KRW-SOL": ["솔라나", "sol"],
}

COMPONENT_LABEL_MAP = {
    "volatility_risk": "변동성",
    "liquidity_risk": "유동성",
    "sentiment_risk": "심리",
    "event_risk": "이벤트",
}


@dataclass(frozen=True)
class HybridRiskInsight:
    asset: str
    risk_level: str
    total_risk_score: float
    top_component: str
    summary: str
    opinion: str
    news_context: str
    operator_actions: list[str]
    latest_headline: str | None
    signal_lines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "risk_level": self.risk_level,
            "total_risk_score": self.total_risk_score,
            "top_component": self.top_component,
            "summary": self.summary,
            "opinion": self.opinion,
            "news_context": self.news_context,
            "operator_actions": self.operator_actions,
            "latest_headline": self.latest_headline,
            "signal_lines": self.signal_lines,
        }


class HybridRiskExplainer:
    """
    Hybrid risk explanation engine for dashboard / alerts / future LLM extension.

    Current design:
    - Rule-based signal interpretation from risk component scores
    - Product-style operational guidance for monitoring / 대응
    - Asset-specific news context integration

    Future extension:
    - OpenAI / external LLM summarization
    - prompt_builder integration
    - multilingual explanation generation
    """

    def __init__(self) -> None:
        self.asset_keyword_map = ASSET_KEYWORD_MAP
        self.component_label_map = COMPONENT_LABEL_MAP

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            score = float(value)
        except Exception:
            return 0.0

        return round(max(0.0, min(score, 1.0)), 4)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    def filter_asset_news(self, news_df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Filter asset-relevant news using title/keyword matching.
        """
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        keywords = self.asset_keyword_map.get(asset, [])
        if not keywords:
            return news_df.head(0).copy()

        title_series = news_df.get("title", pd.Series(dtype=str)).fillna("")
        keyword_series = news_df.get("keyword", pd.Series(dtype=str)).fillna("")

        title_mask = title_series.apply(
            lambda title: any(keyword.lower() in str(title).lower() for keyword in keywords)
        )
        keyword_mask = keyword_series.apply(
            lambda keyword_text: any(token.lower() in str(keyword_text).lower() for token in keywords)
        )

        filtered = news_df.loc[title_mask | keyword_mask].copy()
        if "published_at" in filtered.columns:
            filtered = filtered.sort_values("published_at", ascending=False)
        return filtered

    def _build_component_scores(self, risk_row: pd.Series | dict[str, Any]) -> dict[str, float]:
        return {
            "volatility_risk": self._normalize_score(risk_row.get("volatility_risk", 0.0)),
            "liquidity_risk": self._normalize_score(risk_row.get("liquidity_risk", 0.0)),
            "sentiment_risk": self._normalize_score(risk_row.get("sentiment_risk", 0.0)),
            "event_risk": self._normalize_score(risk_row.get("event_risk", 0.0)),
        }

    def _build_signal_lines(self, component_scores: dict[str, float]) -> list[str]:
        sorted_components = sorted(component_scores.items(), key=lambda item: item[1], reverse=True)
        signal_lines: list[str] = []

        for component_name, score in sorted_components:
            label = self.component_label_map.get(component_name, component_name)
            if score >= 0.7:
                signal_lines.append(f"{label} 리스크가 높습니다 ({score:.2f}).")
            elif score >= 0.4:
                signal_lines.append(f"{label} 리스크가 상승했습니다 ({score:.2f}).")

        if not signal_lines:
            signal_lines.append("주요 리스크 지표는 현재 안정 구간에 가깝습니다.")

        return signal_lines

    def _build_operator_actions(self, component_scores: dict[str, float], risk_level: str) -> list[str]:
        actions: list[str] = []

        if component_scores["liquidity_risk"] >= 0.5:
            actions.append("주문 실행 전 호가 스프레드와 호가 잔량을 우선 확인하세요.")
        if component_scores["event_risk"] >= 0.5:
            actions.append("규제·해킹·거시 이벤트 여부를 추가 검증하고 알림 채널을 유지하세요.")
        if component_scores["volatility_risk"] >= 0.5:
            actions.append("단기 변동성 확대 구간일 수 있으므로 진입 크기와 손절 기준을 보수적으로 설정하세요.")
        if component_scores["sentiment_risk"] >= 0.4:
            actions.append("뉴스와 커뮤니티 심리 악화 여부를 함께 확인해 단기 방향성 왜곡을 경계하세요.")

        if risk_level in {"warning", "critical"}:
            actions.append("실시간 알림을 켜고 리밸런싱 또는 포지션 축소 여부를 검토하세요.")

        if not actions:
            actions.append("현재는 관찰 중심 전략이 유효하며, 다음 사이클의 risk snapshot 변화를 추적하세요.")

        return actions

    def _build_opinion(
        self,
        asset: str,
        risk_level: str,
        top_component_name: str,
        component_scores: dict[str, float],
    ) -> str:
        top_component_label = self.component_label_map.get(top_component_name, top_component_name)
        top_component_score = component_scores[top_component_name]

        if risk_level in {"critical", "warning"}:
            return (
                f"{asset}는 현재 {risk_level} 단계입니다. 단일 지표보다 복합 리스크 대응이 우선이며, "
                f"특히 {top_component_label} 요인이 핵심 압력으로 작동하고 있습니다 ({top_component_score:.2f})."
            )

        if risk_level == "caution":
            return (
                f"{asset}는 현재 caution 단계입니다. 즉시 경보 수준은 아니지만, "
                f"{top_component_label} 중심으로 리스크가 형성되고 있어 모니터링 빈도를 높이는 것이 좋습니다."
            )

        return (
            f"{asset}는 현재 normal 단계입니다. 다만 {top_component_label} 지표가 상대적으로 가장 높기 때문에, "
            "급격한 뉴스 이벤트나 유동성 악화에 대비한 관찰은 유지하는 것이 좋습니다."
        )

    def _build_news_context(self, asset_news_df: pd.DataFrame) -> tuple[str, str | None]:
        if asset_news_df is None or asset_news_df.empty:
            return (
                "현재 자산 특화 뉴스는 제한적입니다. 뉴스 기반 이벤트 리스크보다 시장 미세구조 신호를 우선 해석하는 것이 적절합니다.",
                None,
            )

        latest_headline = self._normalize_text(asset_news_df.iloc[0].get("title", ""))
        news_count = int(len(asset_news_df))
        return (
            f"최근 자산 관련 뉴스 {news_count}건이 반영되었습니다. 가장 최근 헤드라인은 '{latest_headline}' 입니다.",
            latest_headline,
        )

    def explain(
        self,
        risk_row: pd.Series | dict[str, Any],
        news_df: pd.DataFrame | None = None,
    ) -> HybridRiskInsight:
        asset = self._normalize_text(risk_row.get("asset", "UNKNOWN")) or "UNKNOWN"
        risk_level = self._normalize_text(risk_row.get("risk_level", "normal")) or "normal"
        total_risk_score = self._normalize_score(risk_row.get("total_risk_score", 0.0))

        component_scores = self._build_component_scores(risk_row)
        sorted_components = sorted(component_scores.items(), key=lambda item: item[1], reverse=True)
        top_component_name = sorted_components[0][0]

        asset_news_df = self.filter_asset_news(news_df, asset) if news_df is not None else pd.DataFrame()
        signal_lines = self._build_signal_lines(component_scores)
        opinion = self._build_opinion(asset, risk_level, top_component_name, component_scores)
        news_context, latest_headline = self._build_news_context(asset_news_df)
        operator_actions = self._build_operator_actions(component_scores, risk_level)

        return HybridRiskInsight(
            asset=asset,
            risk_level=risk_level,
            total_risk_score=total_risk_score,
            top_component=top_component_name,
            summary=" ".join(signal_lines),
            opinion=opinion,
            news_context=news_context,
            operator_actions=operator_actions,
            latest_headline=latest_headline,
            signal_lines=signal_lines,
        )


explainer = HybridRiskExplainer()


def filter_asset_news(news_df: pd.DataFrame, asset: str) -> pd.DataFrame:
    return explainer.filter_asset_news(news_df, asset)



def generate_hybrid_risk_insight(
    risk_row: pd.Series | dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return explainer.explain(risk_row, news_df).to_dict()