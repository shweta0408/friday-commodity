"""
FRIDAY — data_streamer.py
=========================
class DataStreamer
Handles all live data ingestion:
  • Price OHLCV from yfinance (multi-timeframe)
  • News headlines from Finnhub + NewsAPI fallback
  • Caches data to avoid redundant API calls
"""

import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np

from config import (
    ASSETS, TIMEFRAMES,
    FINNHUB_API_KEY, NEWSAPI_KEY,
    PRICE_REFRESH_SECONDS, NEWS_REFRESH_SECONDS,
)

logger = logging.getLogger("FRIDAY.DataStreamer")


# ─────────────────────────────────────────────────────────────
class PriceCache:
    """Simple TTL cache for price DataFrames."""

    def __init__(self, ttl_seconds: int = PRICE_REFRESH_SECONDS):
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            df, ts = self._cache[key]
            if time.time() - ts < self.ttl:
                return df
        return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        self._cache[key] = (df, time.time())


# ─────────────────────────────────────────────────────────────
class DataStreamer:
    """
    Responsible for:
        1. Fetching multi-timeframe OHLCV data via yfinance
        2. Fetching news headlines from Finnhub / NewsAPI
        3. Returning the latest spot price for each asset
    """

    def __init__(self):
        self._price_cache = PriceCache(ttl_seconds=PRICE_REFRESH_SECONDS)
        self._news_cache:  Dict[str, Tuple[List[dict], float]] = {}
        self._news_ttl = NEWS_REFRESH_SECONDS
        logger.info("DataStreamer initialised.")

    # ── Price Methods ────────────────────────────────────────

    def get_ohlcv(
        self,
        asset_key: str,
        timeframe: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a OHLCV DataFrame for the given asset and timeframe.

        Parameters
        ----------
        asset_key  : one of config.ASSETS keys ("CRUDE_OIL", "GOLD", "SILVER")
        timeframe  : one of config.TIMEFRAMES keys ("15m", "1h", "1d")
        force_refresh : bypass cache

        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{asset_key}_{timeframe}"
        if not force_refresh:
            cached = self._price_cache.get(cache_key)
            if cached is not None:
                return cached

        cfg    = ASSETS[asset_key]
        tf_cfg = TIMEFRAMES[timeframe]
        ticker = cfg["ticker"]

        try:
            df = yf.download(
                ticker,
                period=tf_cfg["period"],
                interval=tf_cfg["interval"],
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning(f"Empty OHLCV for {ticker} @ {timeframe}")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            self._price_cache.set(cache_key, df)
            logger.debug(f"Fetched {len(df)} bars for {ticker} @ {timeframe}")
            return df

        except Exception as exc:
            logger.error(f"yfinance error for {ticker}: {exc}")
            return pd.DataFrame()

    def get_spot_price(self, asset_key: str) -> Optional[float]:
        """Return the latest closing price."""
        df = self.get_ohlcv(asset_key, "15m")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def get_price_change_pct(self, asset_key: str) -> Optional[float]:
        """Return 24h price change percentage."""
        df = self.get_ohlcv(asset_key, "1d")
        if df.empty or len(df) < 2:
            return None
        prev  = float(df["Close"].iloc[-2])
        curr  = float(df["Close"].iloc[-1])
        return round(((curr - prev) / prev) * 100, 3)

    def get_all_ohlcv(self, asset_key: str) -> Dict[str, pd.DataFrame]:
        """Fetch all configured timeframes for a single asset."""
        return {
            tf: self.get_ohlcv(asset_key, tf)
            for tf in TIMEFRAMES
        }

    # ── News Methods ─────────────────────────────────────────

    def get_news(
        self,
        asset_key: str,
        max_articles: int = 20,
        force_refresh: bool = False,
    ) -> List[dict]:
        """
        Returns a list of recent news articles for the given asset.
        Tries Finnhub first; falls back to NewsAPI; falls back to RSS.

        Each article dict:
        {
            "headline": str,
            "summary":  str,
            "source":   str,
            "url":      str,
            "datetime": str (ISO),
        }
        """
        cache_key = asset_key
        if not force_refresh and cache_key in self._news_cache:
            articles, ts = self._news_cache[cache_key]
            if time.time() - ts < self._news_ttl:
                return articles

        articles: List[dict] = []

        # ── Attempt 1: Finnhub ──────────────────────────────
        if FINNHUB_API_KEY and FINNHUB_API_KEY != "demo":
            articles = self._fetch_finnhub_news(asset_key, max_articles)

        # ── Attempt 2: NewsAPI ──────────────────────────────
        if not articles and NEWSAPI_KEY:
            articles = self._fetch_newsapi(asset_key, max_articles)

        # ── Attempt 3: Generic RSS (MarketWatch) ────────────
        if not articles:
            articles = self._fetch_rss_fallback(asset_key)

        self._news_cache[cache_key] = (articles, time.time())
        logger.info(f"Fetched {len(articles)} articles for {asset_key}")
        return articles

    def _fetch_finnhub_news(
        self, asset_key: str, limit: int
    ) -> List[dict]:
        cfg = ASSETS[asset_key]
        today      = datetime.now().strftime("%Y-%m-%d")
        week_ago   = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={cfg['finnhub_symbol']}"
            f"&from={week_ago}&to={today}"
            f"&token={FINNHUB_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if not isinstance(data, list):
                return []
            articles = []
            for item in data[:limit]:
                articles.append({
                    "headline": item.get("headline", ""),
                    "summary":  item.get("summary", ""),
                    "source":   item.get("source", "Finnhub"),
                    "url":      item.get("url", ""),
                    "datetime": datetime.fromtimestamp(
                        item.get("datetime", 0)
                    ).isoformat(),
                })
            return articles
        except Exception as exc:
            logger.warning(f"Finnhub news error: {exc}")
            return []

    def _fetch_newsapi(self, asset_key: str, limit: int) -> List[dict]:
        cfg      = ASSETS[asset_key]
        keywords = " OR ".join(cfg["news_keywords"][:3])
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={keywords}&sortBy=publishedAt&pageSize={limit}"
            f"&language=en&apiKey={NEWSAPI_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "headline": item.get("title", ""),
                    "summary":  item.get("description", ""),
                    "source":   item.get("source", {}).get("name", "NewsAPI"),
                    "url":      item.get("url", ""),
                    "datetime": item.get("publishedAt", ""),
                })
            return articles
        except Exception as exc:
            logger.warning(f"NewsAPI error: {exc}")
            return []

    def _fetch_rss_fallback(self, asset_key: str) -> List[dict]:
        """
        Lightweight RSS fallback — parses MarketWatch commodity feed.
        No external dependency beyond requests.
        """
        rss_urls = {
            "CRUDE_OIL": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
            "GOLD":      "https://feeds.marketwatch.com/marketwatch/marketpulse/",
            "SILVER":    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        }
        url = rss_urls.get(asset_key, "")
        if not url:
            return []
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            text = resp.text
            # Minimal XML extraction without external parser
            articles = []
            items = text.split("<item>")[1:]
            keywords = [k.lower() for k in ASSETS[asset_key]["news_keywords"]]
            for raw in items[:30]:
                title = self._extract_xml_tag(raw, "title")
                desc  = self._extract_xml_tag(raw, "description")
                link  = self._extract_xml_tag(raw, "link")
                pub   = self._extract_xml_tag(raw, "pubDate")
                combined = (title + " " + desc).lower()
                if any(kw in combined for kw in keywords):
                    articles.append({
                        "headline": title,
                        "summary":  desc,
                        "source":   "MarketWatch RSS",
                        "url":      link,
                        "datetime": pub,
                    })
            return articles[:20]
        except Exception as exc:
            logger.warning(f"RSS fallback error: {exc}")
            return []

    @staticmethod
    def _extract_xml_tag(text: str, tag: str) -> str:
        import re
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            content = m.group(1).strip()
            content = re.sub(r"<!\[CDATA\[|\]\]>", "", content)
            return content.strip()
        return ""

    # ── Health / Debug ───────────────────────────────────────

    def health_check(self) -> Dict[str, bool]:
        """Quick connectivity test for each asset."""
        results = {}
        for key in ASSETS:
            df = self.get_ohlcv(key, "1d", force_refresh=True)
            results[key] = not df.empty
        return results
