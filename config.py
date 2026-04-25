# ============================================================
# FRIDAY — config.py
# Central configuration, constants, and environment loader
# ============================================================

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, List

load_dotenv()


# ── API Keys (set in .env or environment) ───────────────────
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "demo")
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "demo")
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

# ── Asset Definitions ────────────────────────────────────────
ASSETS: Dict[str, Dict] = {
    "CRUDE_OIL": {
        "ticker":        "CL=F",
        "display_name":  "Crude Oil (WTI)",
        "finnhub_symbol": "USOIL",
        "news_keywords": ["crude oil", "WTI", "OPEC", "brent", "petroleum", "energy"],
        "unit":          "USD/bbl",
        "emoji":         "🛢️",
    },
    "GOLD": {
        "ticker":        "GC=F",
        "display_name":  "Gold",
        "finnhub_symbol": "GOLD",
        "news_keywords": ["gold", "XAU", "bullion", "precious metals", "safe haven"],
        "unit":          "USD/oz",
        "emoji":         "🥇",
    },
    "SILVER": {
        "ticker":        "SI=F",
        "display_name":  "Silver",
        "finnhub_symbol": "SILVER",
        "news_keywords": ["silver", "XAG", "precious metals", "industrial metals"],
        "unit":          "USD/oz",
        "emoji":         "🥈",
    },
}

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES: Dict[str, Dict] = {
    "15m": {"period": "5d",  "interval": "15m",  "label": "Intraday"},
    "1h":  {"period": "30d", "interval": "1h",   "label": "Swing"},
    "1d":  {"period": "365d", "interval": "1d",   "label": "Trend"},
}

# ── Composite Score Weights ──────────────────────────────────
WEIGHT_SENTIMENT:  float = 0.40
WEIGHT_TECHNICALS: float = 0.40
WEIGHT_WIN_LOSS:   float = 0.20

# ── Trading Signal Thresholds ────────────────────────────────
TRADE_LONG_THRESHOLD:  float = 75.0   # Score > +75 → LONG signal
TRADE_SHORT_THRESHOLD: float = -75.0   # Score < -75 → SHORT signal
RSI_OVERBOUGHT:        float = 70.0   # RSI filter — daily overbought
RSI_OVERSOLD:          float = 30.0   # RSI filter — daily oversold
MIN_WIN_RATE:          float = 0.45   # Confidence penalty below this rate

# ── RSI Divergence Settings ──────────────────────────────────
RSI_PERIOD:           int = 14
DIV_LOOKBACK:         int = 60       # bars to scan for divergence
DIV_MIN_DISTANCE:     int = 5        # min bars between pivot points
DIV_PROMINENCE:       float = 2.0      # scipy peak prominence

# ── Bollinger Band Settings ──────────────────────────────────
BB_PERIOD: int = 20
BB_STD:    float = 2.0

# ── MACD Settings ────────────────────────────────────────────
MACD_FAST:   int = 12
MACD_SLOW:   int = 26
MACD_SIGNAL: int = 9

# ── Volume Profile / S&D Zone Settings ──────────────────────
ZONE_LOOKBACK:  int = 50   # bars for supply/demand detection
ZONE_TOLERANCE: float = 0.005  # 0.5% price band for zone clustering

# ── Sentence Transformer Model ───────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD: float = 0.72   # cosine similarity for "match"

# ── Scheduler Intervals ──────────────────────────────────────
PRICE_REFRESH_SECONDS: int = 60
NEWS_REFRESH_SECONDS:  int = 300

# ── Local Storage Paths ──────────────────────────────────────
DATA_DIR:             str = "friday_data"
TRADE_LOG_PATH:       str = f"{DATA_DIR}/trade_log.json"
HISTORICAL_NEWS_PATH: str = "historical_impact_news.json"   # root-level seed file
VECTOR_INDEX_PATH:    str = f"{DATA_DIR}/news_faiss.index"
VECTOR_META_PATH:     str = f"{DATA_DIR}/news_meta.json"
