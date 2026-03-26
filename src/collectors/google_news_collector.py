from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime, UTC

import requests
import pandas as pd

from src.utils.config import load_config


class GoogleNewsCollector:
    """
    Google News collector using RSS feed.

    - No API key required
    - Stable for MVP
    - Returns structured news records
    """

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        config = load_config(config_path)

        self.keywords: List[str] = config.get("collection", {}).get("news", {}).get("keywords", [])
        self.interval_minutes: int = int(
            config.get("collection", {}).get("news", {}).get("interval_minutes", 15)
        )

    def _request_rss(self, keyword: str) -> str:
        """
        Request Google News RSS feed.
        """
        params = {
            "q": keyword,
            "hl": "ko",
            "gl": "KR",
            "ceid": "KR:ko",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        return response.text

    def _parse_rss(self, xml_text: str, keyword: str) -> List[Dict[str, Any]]:
        """
        Very lightweight RSS parsing (no external dependency).
        """
        items: List[Dict[str, Any]] = []

        # naive parsing for MVP
        split_items = xml_text.split("<item>")[1:]

        for raw_item in split_items:
            try:
                title = self._extract_tag(raw_item, "title")
                link = self._extract_tag(raw_item, "link")
                pub_date = self._extract_tag(raw_item, "pubDate")

                items.append(
                    {
                        "keyword": keyword,
                        "title": title,
                        "link": link,
                        "published_at": pub_date,
                        "collected_at": datetime.now(UTC).isoformat(),
                        "source": "google_news",
                    }
                )
            except Exception:
                continue

        return items

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        """
        Extract simple XML tag content.
        """
        try:
            start = text.index(f"<{tag}>") + len(tag) + 2
            end = text.index(f"</{tag}>")
            return text[start:end]
        except ValueError:
            return ""

    def fetch_by_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Fetch news for a single keyword.
        """
        xml_text = self._request_rss(keyword)
        parsed_items = self._parse_rss(xml_text, keyword)

        df = pd.DataFrame(parsed_items)
        if df.empty:
            return df

        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

        return df

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch news for all configured keywords.
        """
        frames: List[pd.DataFrame] = []

        for keyword in self.keywords:
            try:
                df = self.fetch_by_keyword(keyword)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["title", "link"]).reset_index(drop=True)
        result = result.sort_values("published_at", ascending=False)

        return result


# backward-compatible function

def fetch_google_news(keyword: str) -> List[Dict]:
    """
    Legacy wrapper for compatibility.
    """
    collector = GoogleNewsCollector()
    df = collector.fetch_by_keyword(keyword)

    if df.empty:
        return []

    return df.to_dict(orient="records")


if __name__ == "__main__":
    collector = GoogleNewsCollector()

    df = collector.fetch_all()

    print("\n[GOOGLE NEWS SAMPLE]")
    if not df.empty:
        print(df[["keyword", "title", "published_at"]].head(10))
    else:
        print("No news collected.")