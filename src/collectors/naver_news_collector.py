from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime, UTC

import requests
import pandas as pd

from src.utils.config import load_config


class NaverNewsCollector:
    """
    Naver News collector using open search API (unofficial scraping via query URL).

    - No API key required (HTML-based request)
    - Returns structured news records
    """

    BASE_URL = "https://openapi.naver.com/v1/search/news.json"

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        config = load_config(config_path)

        self.keywords: List[str] = config.get("collection", {}).get("news", {}).get("keywords", [])
        self.interval_minutes: int = int(
            config.get("collection", {}).get("news", {}).get("interval_minutes", 15)
        )

        # NOTE: For MVP, using environment variables is recommended
        import os
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    def _request(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Request Naver News API.
        """
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

        params = {
            "query": keyword,
            "display": 20,
            "sort": "date",
        }

        response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        return data.get("items", [])

    @staticmethod
    def _clean_html(text: str) -> str:
        import re
        import html

        clean = re.sub(r"<.*?>", "", text)
        clean = html.unescape(clean)
        return clean.strip()

    def fetch_by_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Fetch news for a single keyword.
        """
        items = self._request(keyword)

        records: List[Dict[str, Any]] = []

        for item in items:
            try:
                title = self._clean_html(item.get("title", ""))
                link = item.get("link", "")
                pub_date = item.get("pubDate", "")

                records.append(
                    {
                        "keyword": keyword,
                        "title": title,
                        "link": link,
                        "published_at": pub_date,
                        "collected_at": datetime.now(UTC).isoformat(),
                        "source": "naver_news",
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(records)
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

def fetch_naver_news(keyword: str) -> List[Dict]:
    collector = NaverNewsCollector()
    df = collector.fetch_by_keyword(keyword)

    if df.empty:
        return []

    return df.to_dict(orient="records")


if __name__ == "__main__":
    collector = NaverNewsCollector()

    df = collector.fetch_all()

    print("\n[NAVER NEWS SAMPLE]")
    if not df.empty:
        print(df[["keyword", "title", "published_at"]].head(10))
    else:
        print("No news collected.")