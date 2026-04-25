"""
FRIDAY — intelligence_unit.py
==============================
class IntelligenceUnit
Core analytical brain of the FRIDAY system. Contains:

  ┌──────────────────────────────────────────────────────────┐
  │  MODULE A — Technical Analysis Engine                    │
  │   • RSI Divergence Detection (Regular & Hidden)          │
  │   • MACD Signal Extraction                               │
  │   • Bollinger Band Position                              │
  │   • Multi-Timeframe Alignment Score                      │
  │   • Supply / Demand Zone Detection (Volume Profile)      │
  │   • Support / Resistance Identification                  │
  ├──────────────────────────────────────────────────────────┤
  │  MODULE B — Sentiment / NLP Engine                       │
  │   • VADER headline sentiment scoring                     │
  │   • Sentence-Transformer headline embeddings             │
  │   • FAISS vector similarity vs. historical impact news   │
  │   • Composite Sentiment Score (-1 → +1)                  │
  ├──────────────────────────────────────────────────────────┤
  │  MODULE C — Composite Score Engine                       │
  │   • Weighted composite: 40% TA + 40% NLP + 20% W/L      │
  │   • Trade recommendation filter (SMC + Wyckoff)         │
  └──────────────────────────────────────────────────────────┘

Architecture follows Smart Money Concepts (SMC) and
Wyckoff Accumulation/Distribution theory for zone detection.
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks

# Optional heavy imports — degrade gracefully if unavailable
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False
    warnings.warn("sentence-transformers/faiss not installed. Vector search disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    warnings.warn("vaderSentiment not installed. Sentiment will be neutral.")

try:
    import pandas_ta as ta
    _PANDAS_TA = True
except ImportError:
    _PANDAS_TA = False

from config import (
    RSI_PERIOD, DIV_LOOKBACK, DIV_MIN_DISTANCE, DIV_PROMINENCE,
    BB_PERIOD, BB_STD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ZONE_LOOKBACK, ZONE_TOLERANCE,
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
    HISTORICAL_NEWS_PATH, VECTOR_INDEX_PATH, VECTOR_META_PATH,
    RSI_OVERBOUGHT, RSI_OVERSOLD,
    WEIGHT_SENTIMENT, WEIGHT_TECHNICALS, WEIGHT_WIN_LOSS,
    TRADE_LONG_THRESHOLD, TRADE_SHORT_THRESHOLD,
)

logger = logging.getLogger("FRIDAY.IntelligenceUnit")

# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DivergenceSignal:
    """Result of RSI divergence detection on a single timeframe."""
    timeframe:        str
    divergence_type:  str          # "bullish_regular" | "bearish_regular" |
                                   # "bullish_hidden"  | "bearish_hidden"  | "none"
    strength:         float        # 0.0 → 1.0
    bar_index:        int          # Most recent bar where divergence confirmed
    price_at_signal:  float
    rsi_at_signal:    float
    description:      str

@dataclass
class TechnicalSnapshot:
    """Full technical picture for one asset across all timeframes."""
    asset_key:            str
    timestamp:            str
    # RSI Divergences per timeframe
    divergences:          Dict[str, DivergenceSignal] = field(default_factory=dict)
    # Current indicator readings
    rsi_values:           Dict[str, float] = field(default_factory=dict)
    macd_signals:         Dict[str, str]   = field(default_factory=dict)  # "bullish"|"bearish"|"neutral"
    bb_positions:         Dict[str, str]   = field(default_factory=dict)  # "upper"|"lower"|"mid"
    # Supply / Demand zones
    supply_zones:         List[Tuple[float, float]] = field(default_factory=list)  # (lo, hi)
    demand_zones:         List[Tuple[float, float]] = field(default_factory=list)
    # Support / Resistance
    support_levels:       List[float] = field(default_factory=list)
    resistance_levels:    List[float] = field(default_factory=list)
    # Multi-TF alignment
    mtf_alignment_score:  float = 0.0       # -1.0 → +1.0
    # Final TA composite score
    ta_score:             float = 0.0       # -100 → +100
    wyckoff_phase:        str   = "Unknown"

@dataclass
class SentimentSnapshot:
    """NLP analysis result for a set of news headlines."""
    asset_key:          str
    timestamp:          str
    article_count:      int
    vader_score:        float           # -1.0 → +1.0  (compound average)
    impact_matches:     List[dict]      # Matched historical events
    top_impact_move:    float           # Largest matched historical move
    sentiment_score:    float           # Final -1.0 → +1.0
    tone:               str             # "Bullish" | "Bearish" | "Neutral"
    top_headline:       str

@dataclass
class CompositeSignal:
    """Final FRIDAY output for one asset."""
    asset_key:          str
    display_name:       str
    timestamp:          str
    spot_price:         float
    price_change_pct:   float
    sentiment:          SentimentSnapshot
    technicals:         TechnicalSnapshot
    win_loss_ratio:     float           # 0.0 → 1.0
    composite_score:    float           # -100 → +100
    trade_signal:       str             # "LONG" | "SHORT" | "HOLD"
    confidence:         str             # "HIGH" | "MEDIUM" | "LOW"
    reasoning:          List[str]       # Human-readable explanation bullets


# ═══════════════════════════════════════════════════════════════
# INTELLIGENCE UNIT
# ═══════════════════════════════════════════════════════════════

class IntelligenceUnit:
    """
    Core analytical engine for FRIDAY.
    Instantiate once; call analyse(asset_key, ohlcv_dict, news_list, win_loss)
    to obtain a CompositeSignal.
    """

    def __init__(self):
        self._vader   = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None
        self._encoder = None    # lazy-loaded
        self._faiss_index  = None
        self._faiss_meta   = []
        self._hist_news    = []
        self._load_historical_database()
        logger.info("IntelligenceUnit ready.")

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def analyse(
        self,
        asset_key:     str,
        ohlcv_dict:    Dict[str, pd.DataFrame],   # {"15m": df, "1h": df, "1d": df}
        news_articles: List[dict],
        win_loss_ratio: float = 0.50,
        spot_price:    float  = 0.0,
        price_change:  float  = 0.0,
    ) -> CompositeSignal:
        """
        Full pipeline: TA → Sentiment → Composite Score → Signal.
        """
        from datetime import datetime
        ts = datetime.now().isoformat(timespec="seconds")

        # ── 1. Technical Analysis ──────────────────────────
        tech = self._run_technical_analysis(asset_key, ohlcv_dict, ts)

        # ── 2. Sentiment Analysis ──────────────────────────
        sent = self._run_sentiment_analysis(asset_key, news_articles, ts)

        # ── 3. Composite Score ────────────────────────────
        signal = self._build_composite_signal(
            asset_key, ts, spot_price, price_change,
            tech, sent, win_loss_ratio
        )
        return signal

    # ══════════════════════════════════════════════════════════
    # MODULE A — TECHNICAL ANALYSIS ENGINE
    # ══════════════════════════════════════════════════════════

    def _run_technical_analysis(
        self,
        asset_key: str,
        ohlcv_dict: Dict[str, pd.DataFrame],
        ts: str,
    ) -> TechnicalSnapshot:
        snap = TechnicalSnapshot(asset_key=asset_key, timestamp=ts)

        ta_scores = []

        for tf, df in ohlcv_dict.items():
            if df is None or df.empty or len(df) < RSI_PERIOD + 10:
                logger.warning(f"Insufficient data for {asset_key} @ {tf}")
                continue

            close  = df["Close"].astype(float)
            high   = df["High"].astype(float)
            low    = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            # ── RSI ──────────────────────────────────────
            rsi_series = self._calc_rsi(close, RSI_PERIOD)
            snap.rsi_values[tf] = round(float(rsi_series.iloc[-1]), 2)

            # ── RSI Divergence ───────────────────────────
            div_signal = self._detect_rsi_divergence(
                close, rsi_series, tf, float(close.iloc[-1])
            )
            snap.divergences[tf] = div_signal

            # ── MACD ─────────────────────────────────────
            macd_sig = self._calc_macd_signal(close)
            snap.macd_signals[tf] = macd_sig

            # ── Bollinger Bands ───────────────────────────
            bb_pos = self._calc_bb_position(close)
            snap.bb_positions[tf] = bb_pos

            # ── Per-TF score contribution ─────────────────
            tf_score = self._score_timeframe(div_signal, macd_sig, bb_pos, rsi_series)
            ta_scores.append((tf, tf_score))

        # ── Supply & Demand (on 1h chart) ────────────────
        if "1h" in ohlcv_dict and not ohlcv_dict["1h"].empty:
            df1h = ohlcv_dict["1h"]
            snap.supply_zones, snap.demand_zones = self._detect_sd_zones(
                df1h["High"].astype(float),
                df1h["Low"].astype(float),
                df1h["Volume"].astype(float),
            )

        # ── Support / Resistance (on 1d chart) ───────────
        if "1d" in ohlcv_dict and not ohlcv_dict["1d"].empty:
            df1d = ohlcv_dict["1d"]
            snap.support_levels, snap.resistance_levels = (
                self._detect_support_resistance(
                    df1d["High"].astype(float),
                    df1d["Low"].astype(float),
                    df1d["Close"].astype(float),
                )
            )

        # ── Multi-TF Alignment ────────────────────────────
        snap.mtf_alignment_score = self._multi_tf_alignment(
            snap.divergences, snap.macd_signals, snap.demand_zones, snap.supply_zones
        )

        # ── Wyckoff Phase ─────────────────────────────────
        if "1d" in ohlcv_dict and not ohlcv_dict["1d"].empty:
            snap.wyckoff_phase = self._detect_wyckoff_phase(
                ohlcv_dict["1d"]["Close"].astype(float),
                ohlcv_dict["1d"]["Volume"].astype(float),
            )

        # ── Final TA Score ────────────────────────────────
        snap.ta_score = self._aggregate_ta_score(ta_scores, snap.mtf_alignment_score)
        return snap

    # ── RSI Calculation ──────────────────────────────────────

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """Wilder's smoothed RSI."""
        delta  = close.diff()
        gain   = delta.where(delta > 0, 0.0)
        loss   = (-delta).where(delta < 0, 0.0)
        avg_g  = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_l  = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs     = avg_g / avg_l.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    # ── RSI Divergence Detection ─────────────────────────────

    def _detect_rsi_divergence(
        self,
        close:    pd.Series,
        rsi:      pd.Series,
        timeframe: str,
        current_price: float,
    ) -> DivergenceSignal:
        """
        Detect Regular and Hidden RSI Divergence using scipy peak detection.

        Regular Bullish  : Price makes lower low  → RSI makes higher low  (reversal up)
        Regular Bearish  : Price makes higher high → RSI makes lower high  (reversal dn)
        Hidden  Bullish  : Price makes higher low  → RSI makes lower low   (continuation up)
        Hidden  Bearish  : Price makes lower high  → RSI makes higher high (continuation dn)
        """
        lookback = min(DIV_LOOKBACK, len(close) - 1)
        c   = close.iloc[-lookback:].values.astype(float)
        r   = rsi.iloc[-lookback:].values.astype(float)
        n   = len(c)
        idx = len(close) - 1  # absolute bar index

        if n < DIV_MIN_DISTANCE * 2 + 2:
            return self._no_divergence(timeframe, idx, current_price, float(rsi.iloc[-1]))

        # ── Find price pivots ──────────────────────────────
        price_lows,  _ = find_peaks(-c, distance=DIV_MIN_DISTANCE, prominence=DIV_PROMINENCE)
        price_highs, _ = find_peaks( c, distance=DIV_MIN_DISTANCE, prominence=DIV_PROMINENCE)
        rsi_lows,    _ = find_peaks(-r, distance=DIV_MIN_DISTANCE, prominence=DIV_PROMINENCE)
        rsi_highs,   _ = find_peaks( r, distance=DIV_MIN_DISTANCE, prominence=DIV_PROMINENCE)

        current_rsi = float(rsi.iloc[-1])

        # ── Regular Bullish: price LL, RSI HL ─────────────
        rb = self._check_bullish_regular(c, r, price_lows, rsi_lows, n)
        if rb > 0:
            return DivergenceSignal(
                timeframe=timeframe, divergence_type="bullish_regular",
                strength=min(1.0, rb), bar_index=idx,
                price_at_signal=current_price, rsi_at_signal=current_rsi,
                description=f"[{timeframe}] Regular Bullish Div — price LL, RSI HL (str={rb:.2f})"
            )

        # ── Regular Bearish: price HH, RSI LH ─────────────
        rbe = self._check_bearish_regular(c, r, price_highs, rsi_highs, n)
        if rbe > 0:
            return DivergenceSignal(
                timeframe=timeframe, divergence_type="bearish_regular",
                strength=min(1.0, rbe), bar_index=idx,
                price_at_signal=current_price, rsi_at_signal=current_rsi,
                description=f"[{timeframe}] Regular Bearish Div — price HH, RSI LH (str={rbe:.2f})"
            )

        # ── Hidden Bullish: price HL, RSI LL ──────────────
        hb = self._check_bullish_hidden(c, r, price_lows, rsi_lows, n)
        if hb > 0:
            return DivergenceSignal(
                timeframe=timeframe, divergence_type="bullish_hidden",
                strength=min(1.0, hb), bar_index=idx,
                price_at_signal=current_price, rsi_at_signal=current_rsi,
                description=f"[{timeframe}] Hidden Bullish Div — price HL, RSI LL (str={hb:.2f})"
            )

        # ── Hidden Bearish: price LH, RSI HH ──────────────
        hbe = self._check_bearish_hidden(c, r, price_highs, rsi_highs, n)
        if hbe > 0:
            return DivergenceSignal(
                timeframe=timeframe, divergence_type="bearish_hidden",
                strength=min(1.0, hbe), bar_index=idx,
                price_at_signal=current_price, rsi_at_signal=current_rsi,
                description=f"[{timeframe}] Hidden Bearish Div — price LH, RSI HH (str={hbe:.2f})"
            )

        return self._no_divergence(timeframe, idx, current_price, current_rsi)

    # ── Divergence sub-checks ────────────────────────────────

    @staticmethod
    def _check_bullish_regular(
        c: np.ndarray, r: np.ndarray,
        p_lows: np.ndarray, r_lows: np.ndarray, n: int
    ) -> float:
        """
        Regular Bullish: most-recent price low < previous price low
                         most-recent RSI  low > previous RSI  low
        Returns strength 0.0–1.0 (0 = no divergence).
        """
        if len(p_lows) < 2 or len(r_lows) < 2:
            return 0.0
        # Take two most recent lows within the last 1/3 of the window
        recent_cutoff = int(n * 0.6)
        p_recent = [i for i in p_lows if i >= recent_cutoff]
        r_recent = [i for i in r_lows if i >= recent_cutoff]
        if len(p_recent) < 2 or len(r_recent) < 2:
            return 0.0
        p1, p2 = p_recent[-2], p_recent[-1]   # older, newer
        # Find closest RSI lows to price low positions
        rr1 = min(r_recent, key=lambda x: abs(x - p1))
        rr2 = min(r_recent, key=lambda x: abs(x - p2))
        if rr1 == rr2:
            return 0.0
        price_ll = c[p2] < c[p1]
        rsi_hl   = r[rr2] > r[rr1]
        if price_ll and rsi_hl:
            # Strength = RSI divergence magnitude / price divergence magnitude
            price_drop = (c[p1] - c[p2]) / (c[p1] + 1e-9)
            rsi_rise   = (r[rr2] - r[rr1]) / (r[rr1] + 1e-9)
            strength   = np.clip((rsi_rise + price_drop) / 2, 0.05, 1.0)
            return float(strength)
        return 0.0

    @staticmethod
    def _check_bearish_regular(
        c: np.ndarray, r: np.ndarray,
        p_highs: np.ndarray, r_highs: np.ndarray, n: int
    ) -> float:
        """Regular Bearish: price HH, RSI LH."""
        if len(p_highs) < 2 or len(r_highs) < 2:
            return 0.0
        recent_cutoff = int(n * 0.6)
        p_recent = [i for i in p_highs if i >= recent_cutoff]
        r_recent = [i for i in r_highs if i >= recent_cutoff]
        if len(p_recent) < 2 or len(r_recent) < 2:
            return 0.0
        p1, p2 = p_recent[-2], p_recent[-1]
        rr1 = min(r_recent, key=lambda x: abs(x - p1))
        rr2 = min(r_recent, key=lambda x: abs(x - p2))
        if rr1 == rr2:
            return 0.0
        price_hh = c[p2] > c[p1]
        rsi_lh   = r[rr2] < r[rr1]
        if price_hh and rsi_lh:
            price_rise = (c[p2] - c[p1]) / (c[p1] + 1e-9)
            rsi_drop   = (r[rr1] - r[rr2]) / (r[rr1] + 1e-9)
            strength   = np.clip((rsi_drop + price_rise) / 2, 0.05, 1.0)
            return float(strength)
        return 0.0

    @staticmethod
    def _check_bullish_hidden(
        c: np.ndarray, r: np.ndarray,
        p_lows: np.ndarray, r_lows: np.ndarray, n: int
    ) -> float:
        """Hidden Bullish: price HL, RSI LL (continuation of uptrend)."""
        if len(p_lows) < 2 or len(r_lows) < 2:
            return 0.0
        recent_cutoff = int(n * 0.6)
        p_recent = [i for i in p_lows if i >= recent_cutoff]
        r_recent = [i for i in r_lows if i >= recent_cutoff]
        if len(p_recent) < 2 or len(r_recent) < 2:
            return 0.0
        p1, p2 = p_recent[-2], p_recent[-1]
        rr1 = min(r_recent, key=lambda x: abs(x - p1))
        rr2 = min(r_recent, key=lambda x: abs(x - p2))
        if rr1 == rr2:
            return 0.0
        price_hl = c[p2] > c[p1]   # higher low
        rsi_ll   = r[rr2] < r[rr1]  # lower low
        if price_hl and rsi_ll:
            strength = np.clip(abs(r[rr1] - r[rr2]) / 20, 0.05, 0.8)
            return float(strength)
        return 0.0

    @staticmethod
    def _check_bearish_hidden(
        c: np.ndarray, r: np.ndarray,
        p_highs: np.ndarray, r_highs: np.ndarray, n: int
    ) -> float:
        """Hidden Bearish: price LH, RSI HH (continuation of downtrend)."""
        if len(p_highs) < 2 or len(r_highs) < 2:
            return 0.0
        recent_cutoff = int(n * 0.6)
        p_recent = [i for i in p_highs if i >= recent_cutoff]
        r_recent = [i for i in r_highs if i >= recent_cutoff]
        if len(p_recent) < 2 or len(r_recent) < 2:
            return 0.0
        p1, p2 = p_recent[-2], p_recent[-1]
        rr1 = min(r_recent, key=lambda x: abs(x - p1))
        rr2 = min(r_recent, key=lambda x: abs(x - p2))
        if rr1 == rr2:
            return 0.0
        price_lh = c[p2] < c[p1]   # lower high
        rsi_hh   = r[rr2] > r[rr1]  # higher high
        if price_lh and rsi_hh:
            strength = np.clip(abs(r[rr2] - r[rr1]) / 20, 0.05, 0.8)
            return float(strength)
        return 0.0

    @staticmethod
    def _no_divergence(tf, idx, price, rsi_val) -> DivergenceSignal:
        return DivergenceSignal(
            timeframe=tf, divergence_type="none", strength=0.0,
            bar_index=idx, price_at_signal=price, rsi_at_signal=rsi_val,
            description=f"[{tf}] No divergence detected"
        )

    # ── MACD ─────────────────────────────────────────────────

    @staticmethod
    def _calc_macd_signal(close: pd.Series) -> str:
        ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist        = macd_line - signal_line
        if len(hist) < 2:
            return "neutral"
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            return "bullish_cross"
        if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            return "bearish_cross"
        if hist.iloc[-1] > 0:
            return "bullish"
        if hist.iloc[-1] < 0:
            return "bearish"
        return "neutral"

    # ── Bollinger Bands ───────────────────────────────────────

    @staticmethod
    def _calc_bb_position(close: pd.Series) -> str:
        sma   = close.rolling(BB_PERIOD).mean()
        std   = close.rolling(BB_PERIOD).std()
        upper = sma + BB_STD * std
        lower = sma - BB_STD * std
        c     = float(close.iloc[-1])
        u     = float(upper.iloc[-1])
        l     = float(lower.iloc[-1])
        m     = float(sma.iloc[-1])
        if c >= u:
            return "upper"
        if c <= l:
            return "lower"
        if c > m:
            return "upper_mid"
        return "lower_mid"

    # ── Supply & Demand Zone Detection (SMC / Wyckoff) ───────

    def _detect_sd_zones(
        self,
        high: pd.Series, low: pd.Series, volume: pd.Series,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Supply/Demand zones via volume-weighted pivot clustering.

        Wyckoff logic:
        • Supply zone = area of high-volume rejection from above
        • Demand zone = area of high-volume reversal from below

        Returns (supply_zones, demand_zones) where each zone = (low_price, high_price)
        """
        n = min(ZONE_LOOKBACK, len(high))
        h = high.iloc[-n:].values
        l = low.iloc[-n:].values
        v = volume.iloc[-n:].values

        # Normalise volume to [0, 1]
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)

        # Pivot highs (potential supply zones)
        ph_idx = argrelextrema(h, np.greater, order=3)[0]
        # Pivot lows  (potential demand zones)
        pl_idx = argrelextrema(l, np.less,    order=3)[0]

        supply_zones: List[Tuple[float, float]] = []
        demand_zones: List[Tuple[float, float]] = []

        # Filter by volume significance (> 60th percentile)
        vol_threshold = np.percentile(v_norm, 60)

        for i in ph_idx:
            if v_norm[i] >= vol_threshold:
                zone_lo = float(h[i] * (1 - ZONE_TOLERANCE))
                zone_hi = float(h[i] * (1 + ZONE_TOLERANCE))
                supply_zones.append((zone_lo, zone_hi))

        for i in pl_idx:
            if v_norm[i] >= vol_threshold:
                zone_lo = float(l[i] * (1 - ZONE_TOLERANCE))
                zone_hi = float(l[i] * (1 + ZONE_TOLERANCE))
                demand_zones.append((zone_lo, zone_hi))

        # Merge overlapping zones
        supply_zones = self._merge_zones(supply_zones)
        demand_zones = self._merge_zones(demand_zones)

        # Keep the 5 most recent / significant
        return supply_zones[-5:], demand_zones[-5:]

    @staticmethod
    def _merge_zones(
        zones: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge overlapping price zones."""
        if not zones:
            return []
        zones_sorted = sorted(zones, key=lambda x: x[0])
        merged = [zones_sorted[0]]
        for lo, hi in zones_sorted[1:]:
            prev_lo, prev_hi = merged[-1]
            if lo <= prev_hi:   # overlap
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    # ── Support & Resistance ─────────────────────────────────

    @staticmethod
    def _detect_support_resistance(
        high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> Tuple[List[float], List[float]]:
        """
        Identify key support and resistance levels from daily chart.
        Uses local extrema with minimum distance filtering.
        """
        c = close.values.astype(float)
        h = high.values.astype(float)
        l = low.values.astype(float)

        resistance_idx = argrelextrema(h, np.greater, order=5)[0]
        support_idx    = argrelextrema(l, np.less,    order=5)[0]

        resistance = sorted(set(round(h[i], 2) for i in resistance_idx), reverse=True)[:5]
        support    = sorted(set(round(l[i], 2) for i in support_idx))[:5]
        return support, resistance

    # ── Multi-Timeframe Alignment ─────────────────────────────

    def _multi_tf_alignment(
        self,
        divergences:   Dict[str, DivergenceSignal],
        macd_signals:  Dict[str, str],
        demand_zones:  List[Tuple[float, float]],
        supply_zones:  List[Tuple[float, float]],
    ) -> float:
        """
        Returns alignment score from -1.0 (fully bearish aligned) to +1.0
        (fully bullish aligned) across timeframes.

        SMC Rule: 15m bullish divergence inside 1h demand zone = strong signal.
        """
        bullish_votes = 0.0
        bearish_votes = 0.0
        total_weight  = 0.0

        tf_weights = {"15m": 0.3, "1h": 0.4, "1d": 0.3}

        for tf, weight in tf_weights.items():
            div = divergences.get(tf)
            mac = macd_signals.get(tf, "neutral")

            if div:
                if "bullish" in div.divergence_type:
                    bullish_votes += weight * (0.5 + div.strength * 0.5)
                elif "bearish" in div.divergence_type:
                    bearish_votes += weight * (0.5 + div.strength * 0.5)

            if "bullish" in mac:
                bullish_votes += weight * 0.3
            elif "bearish" in mac:
                bearish_votes += weight * 0.3

            total_weight += weight

        # SMC confluence bonus: 15m bullish div + demand zone
        if (
            divergences.get("15m") and
            "bullish" in divergences["15m"].divergence_type and
            demand_zones
        ):
            price = divergences["15m"].price_at_signal
            in_zone = any(lo <= price <= hi for lo, hi in demand_zones)
            if in_zone:
                bullish_votes += 0.25   # Strong SMC confluence bonus

        # SMC confluence: 15m bearish div + supply zone
        if (
            divergences.get("15m") and
            "bearish" in divergences["15m"].divergence_type and
            supply_zones
        ):
            price = divergences["15m"].price_at_signal
            in_zone = any(lo <= price <= hi for lo, hi in supply_zones)
            if in_zone:
                bearish_votes += 0.25

        net = bullish_votes - bearish_votes
        return float(np.clip(net, -1.0, 1.0))

    # ── Per-Timeframe Score ───────────────────────────────────

    @staticmethod
    def _score_timeframe(
        div:     DivergenceSignal,
        macd:    str,
        bb_pos:  str,
        rsi_s:   pd.Series,
    ) -> float:
        """Return a -1.0 → +1.0 score for one timeframe."""
        score = 0.0
        rsi_val = float(rsi_s.iloc[-1])

        # Divergence (50% weight)
        if "bullish" in div.divergence_type:
            score += 0.5 * div.strength * (1.2 if "regular" in div.divergence_type else 0.9)
        elif "bearish" in div.divergence_type:
            score -= 0.5 * div.strength * (1.2 if "regular" in div.divergence_type else 0.9)

        # MACD (30% weight)
        macd_map = {
            "bullish_cross": 0.30,
            "bullish":       0.15,
            "bearish_cross": -0.30,
            "bearish":       -0.15,
            "neutral":        0.0,
        }
        score += macd_map.get(macd, 0.0)

        # Bollinger (20% weight)
        bb_map = {
            "lower":     0.20,   # price at lower band → mean reversion potential
            "lower_mid": 0.05,
            "upper":    -0.20,
            "upper_mid":-0.05,
        }
        score += bb_map.get(bb_pos, 0.0)

        # RSI extreme penalty / bonus
        if rsi_val < RSI_OVERSOLD:
            score += 0.10    # oversold → bullish lean
        elif rsi_val > RSI_OVERBOUGHT:
            score -= 0.10    # overbought → bearish lean

        return float(np.clip(score, -1.0, 1.0))

    # ── Aggregate TA Score ────────────────────────────────────

    @staticmethod
    def _aggregate_ta_score(
        tf_scores: List[Tuple[str, float]],
        mtf_alignment: float,
    ) -> float:
        """
        Weighted average of per-timeframe scores + MTF alignment bonus.
        Output: -100 → +100
        """
        if not tf_scores:
            return 0.0
        tf_weights = {"15m": 0.30, "1h": 0.40, "1d": 0.30}
        weighted_sum = 0.0
        total_w      = 0.0
        for tf, score in tf_scores:
            w = tf_weights.get(tf, 0.33)
            weighted_sum += score * w
            total_w      += w
        if total_w == 0:
            return 0.0
        base = weighted_sum / total_w
        # MTF alignment adds up to ±10 points
        final = (base + mtf_alignment * 0.1) * 100
        return float(np.clip(final, -100, 100))

    # ── Wyckoff Phase Detection ───────────────────────────────

    @staticmethod
    def _detect_wyckoff_phase(close: pd.Series, volume: pd.Series) -> str:
        """
        Simplified Wyckoff phase detection based on price trend + volume:

        Accumulation  : Downtrend slowing, rising volume on recoveries
        Distribution  : Uptrend slowing, rising volume on declines
        Mark Up       : Uptrend with expanding volume
        Mark Down     : Downtrend with expanding volume
        """
        if len(close) < 30:
            return "Unknown"
        c  = close.values[-30:]
        v  = volume.values[-30:]
        v  = (v - v.min()) / (v.max() - v.min() + 1e-9)

        trend_slope = np.polyfit(range(len(c)), c, 1)[0]
        # Split into two halves
        h1_v = v[:15].mean()
        h2_v = v[15:].mean()
        vol_accel = h2_v - h1_v

        if trend_slope > 0 and vol_accel > 0.05:
            return "Mark Up (Wyckoff)"
        if trend_slope < 0 and vol_accel > 0.05:
            return "Mark Down (Wyckoff)"
        if trend_slope < 0 and vol_accel < 0:
            return "Accumulation (Wyckoff)"
        if trend_slope > 0 and vol_accel < 0:
            return "Distribution (Wyckoff)"
        return "Consolidation (Wyckoff)"

    # ══════════════════════════════════════════════════════════
    # MODULE B — SENTIMENT / NLP ENGINE
    # ══════════════════════════════════════════════════════════

    def _run_sentiment_analysis(
        self,
        asset_key: str,
        articles: List[dict],
        ts: str,
    ) -> SentimentSnapshot:
        from datetime import datetime

        if not articles:
            return SentimentSnapshot(
                asset_key=asset_key, timestamp=ts,
                article_count=0, vader_score=0.0,
                impact_matches=[], top_impact_move=0.0,
                sentiment_score=0.0, tone="Neutral",
                top_headline="No recent news found."
            )

        headlines = [a.get("headline", "") for a in articles if a.get("headline")]

        # ── VADER Scoring ─────────────────────────────────
        vader_scores = [self._score_headline_vader(h) for h in headlines]
        avg_vader    = float(np.mean(vader_scores)) if vader_scores else 0.0

        # ── Vector Similarity vs Historical Impacts ────────
        impact_matches, top_move = self._find_impact_events(
            headlines, asset_key
        )

        # ── Composite Sentiment Score ──────────────────────
        # Base: VADER (60%) + Impact Event adjustment (40%)
        impact_adj = 0.0
        if impact_matches:
            # Scale: a 10% historical move ↔ ±0.5 adjustment
            best = impact_matches[0]
            direction  = 1.0 if best["direction"] == "bullish" else -1.0
            similarity = best["similarity"]
            impact_adj = direction * similarity * min(abs(best["move_pct"]) / 20.0, 1.0)

        sent_score = 0.6 * avg_vader + 0.4 * impact_adj
        sent_score = float(np.clip(sent_score, -1.0, 1.0))

        if sent_score >  0.15:
            tone = "Bullish"
        elif sent_score < -0.15:
            tone = "Bearish"
        else:
            tone = "Neutral"

        top_headline = headlines[0] if headlines else ""

        return SentimentSnapshot(
            asset_key=asset_key, timestamp=ts,
            article_count=len(articles),
            vader_score=avg_vader,
            impact_matches=impact_matches[:3],
            top_impact_move=top_move,
            sentiment_score=sent_score,
            tone=tone,
            top_headline=top_headline,
        )

    # ── VADER Headline Scoring ────────────────────────────────

    def _score_headline_vader(self, headline: str) -> float:
        """Returns VADER compound score (-1 to +1)."""
        if not self._vader or not headline:
            return 0.0
        result = self._vader.polarity_scores(headline)
        return float(result["compound"])

    # ── Vector Embedding & FAISS Search ─────────────────────

    def _get_encoder(self):
        """Lazy-load sentence transformer to avoid startup delay."""
        if self._encoder is None and _VECTOR_AVAILABLE:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            self._build_faiss_index()
        return self._encoder

    def _load_historical_database(self):
        """Load historical impact news from JSON seed file."""
        try:
            with open(HISTORICAL_NEWS_PATH, "r") as f:
                self._hist_news = json.load(f)
            logger.info(f"Loaded {len(self._hist_news)} historical impact events.")
        except FileNotFoundError:
            logger.warning("Historical news database not found. Impact matching disabled.")
            self._hist_news = []

    def _build_faiss_index(self):
        """
        Build FAISS flat L2 index from historical news embeddings.
        Called once on first use. Index stored in memory (and optionally disk).
        """
        if not self._hist_news or not _VECTOR_AVAILABLE:
            return

        headlines = [item["headline"] for item in self._hist_news]
        logger.info("Encoding historical news headlines into vector embeddings...")
        embeddings = self._encoder.encode(headlines, show_progress_bar=False)
        embeddings = embeddings.astype("float32")

        # Normalise for cosine similarity (inner product ≡ cosine after normalisation)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)   # Inner Product = cosine sim
        self._faiss_index.add(embeddings)
        self._faiss_meta = self._hist_news
        logger.info(f"FAISS index built with {self._faiss_index.ntotal} vectors (dim={dim}).")

    def _find_impact_events(
        self,
        headlines: List[str],
        asset_key: str,
        top_k: int = 3,
    ) -> Tuple[List[dict], float]:
        """
        Core vector search method.
        For each current headline, find the most similar historical
        high-impact events using cosine similarity (FAISS).

        Returns:
            matched_events : list of match dicts with similarity scores
            top_move_pct   : the largest historical move matched
        """
        encoder = self._get_encoder()
        if encoder is None or self._faiss_index is None or not headlines:
            return self._keyword_fallback(headlines, asset_key)

        # Encode current headlines
        query_emb = encoder.encode(headlines[:10], show_progress_bar=False)
        query_emb = query_emb.astype("float32")
        faiss.normalize_L2(query_emb)

        # Search FAISS index
        similarities, indices = self._faiss_index.search(query_emb, top_k)

        matched: Dict[str, dict] = {}   # deduplicate by historical news id

        for q_idx, (sim_row, idx_row) in enumerate(zip(similarities, indices)):
            for sim, hist_idx in zip(sim_row, idx_row):
                if hist_idx < 0 or hist_idx >= len(self._faiss_meta):
                    continue
                sim_val = float(sim)
                if sim_val < SIMILARITY_THRESHOLD:
                    continue
                hist_item = self._faiss_meta[hist_idx]
                # Only match same asset
                if hist_item.get("asset") != asset_key:
                    continue
                event_id = hist_item["id"]
                if event_id not in matched or matched[event_id]["similarity"] < sim_val:
                    matched[event_id] = {
                        "id":             event_id,
                        "historical_headline": hist_item["headline"],
                        "current_headline":    headlines[q_idx],
                        "similarity":          round(sim_val, 4),
                        "move_pct":            hist_item["move_pct"],
                        "direction":           hist_item["direction"],
                        "date":                hist_item["date"],
                    }

        results = sorted(matched.values(), key=lambda x: x["similarity"], reverse=True)

        top_move = max((abs(r["move_pct"]) for r in results), default=0.0)
        logger.info(f"Vector search found {len(results)} impact match(es) for {asset_key}")
        return results, top_move

    def _keyword_fallback(
        self, headlines: List[str], asset_key: str
    ) -> Tuple[List[dict], float]:
        """
        Simple keyword overlap fallback when sentence-transformers is unavailable.
        Uses Jaccard similarity on word sets.
        """
        matches = []
        asset_hist = [h for h in self._hist_news if h.get("asset") == asset_key]
        for h_item in asset_hist:
            hist_words = set(h_item["headline"].lower().split())
            for cur_hl in headlines[:10]:
                cur_words = set(cur_hl.lower().split())
                if not cur_words:
                    continue
                jaccard = len(hist_words & cur_words) / len(hist_words | cur_words)
                if jaccard >= 0.2:  # lower threshold for keyword fallback
                    matches.append({
                        "id":             h_item["id"],
                        "historical_headline": h_item["headline"],
                        "current_headline":    cur_hl,
                        "similarity":          round(jaccard, 4),
                        "move_pct":            h_item["move_pct"],
                        "direction":           h_item["direction"],
                        "date":                h_item["date"],
                    })
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_move = max((abs(m["move_pct"]) for m in matches), default=0.0)
        return matches[:3], top_move

    # ══════════════════════════════════════════════════════════
    # MODULE C — COMPOSITE SCORE ENGINE
    # ══════════════════════════════════════════════════════════

    def _build_composite_signal(
        self,
        asset_key:      str,
        ts:             str,
        spot_price:     float,
        price_change:   float,
        tech:           TechnicalSnapshot,
        sent:           SentimentSnapshot,
        win_loss_ratio: float,
    ) -> CompositeSignal:
        from config import ASSETS

        asset_cfg = ASSETS[asset_key]

        # ── Normalise inputs to -100 → +100 ──────────────
        ta_score   = tech.ta_score                           # already -100 → +100
        sent_score = sent.sentiment_score * 100              # -1→+1 scaled to -100→+100
        wl_score   = (win_loss_ratio - 0.5) * 200           # 0→1 mapped to -100→+100

        # ── Win/Loss confidence penalty ───────────────────
        from config import MIN_WIN_RATE
        wl_confidence_factor = 1.0
        if win_loss_ratio < MIN_WIN_RATE:
            # Reduce TA and sentiment scores by up to 30%
            penalty = (MIN_WIN_RATE - win_loss_ratio) / MIN_WIN_RATE
            wl_confidence_factor = 1.0 - (penalty * 0.30)

        # ── Weighted composite ────────────────────────────
        raw_score = (
            WEIGHT_SENTIMENT  * sent_score * wl_confidence_factor +
            WEIGHT_TECHNICALS * ta_score   * wl_confidence_factor +
            WEIGHT_WIN_LOSS   * wl_score
        )
        composite = float(np.clip(raw_score, -100, 100))

        # ── Trade Signal Filter ───────────────────────────
        daily_rsi = tech.rsi_values.get("1d", 50.0)

        reasoning: List[str] = []
        trade_signal = "HOLD"

        if composite >= TRADE_LONG_THRESHOLD:
            if daily_rsi > RSI_OVERBOUGHT:
                trade_signal = "HOLD"
                reasoning.append(
                    f"⚠️ Score={composite:.1f} triggers LONG but 1D RSI={daily_rsi:.1f} is overbought — HOLD"
                )
            else:
                trade_signal = "LONG"
                reasoning.append(f"✅ Composite score {composite:.1f} exceeds +{TRADE_LONG_THRESHOLD} → LONG")

        elif composite <= TRADE_SHORT_THRESHOLD:
            if daily_rsi < RSI_OVERSOLD:
                trade_signal = "HOLD"
                reasoning.append(
                    f"⚠️ Score={composite:.1f} triggers SHORT but 1D RSI={daily_rsi:.1f} is oversold — HOLD"
                )
            else:
                trade_signal = "SHORT"
                reasoning.append(f"✅ Composite score {composite:.1f} below -{abs(TRADE_SHORT_THRESHOLD)} → SHORT")

        else:
            reasoning.append(
                f"🟡 Score={composite:.1f} — insufficient conviction (threshold ±{TRADE_LONG_THRESHOLD})"
            )

        # ── Divergence reasoning bullets ──────────────────
        for tf, div in tech.divergences.items():
            if div.divergence_type != "none":
                reasoning.append(f"📊 {div.description}")

        # ── SMC confluence bullets ─────────────────────────
        if tech.demand_zones:
            reasoning.append(
                f"🟢 {len(tech.demand_zones)} demand zone(s) identified at "
                f"{tech.demand_zones[-1][0]:.2f}–{tech.demand_zones[-1][1]:.2f}"
            )
        if tech.supply_zones:
            reasoning.append(
                f"🔴 {len(tech.supply_zones)} supply zone(s) identified at "
                f"{tech.supply_zones[-1][0]:.2f}–{tech.supply_zones[-1][1]:.2f}"
            )

        # ── Wyckoff ───────────────────────────────────────
        reasoning.append(f"📈 Wyckoff: {tech.wyckoff_phase}")

        # ── Sentiment reasoning ───────────────────────────
        if sent.impact_matches:
            m = sent.impact_matches[0]
            reasoning.append(
                f"📰 Impact event matched ({m['similarity']*100:.0f}% sim): "
                f"'{m['historical_headline'][:60]}...' "
                f"→ historical move {m['move_pct']:+.1f}%"
            )
        reasoning.append(
            f"🗞️ Sentiment: {sent.tone} (VADER={sent.vader_score:+.3f}, "
            f"score={sent.sentiment_score:+.3f})"
        )

        # ── Confidence tier ───────────────────────────────
        abs_score = abs(composite)
        if abs_score >= 85:
            confidence = "HIGH"
        elif abs_score >= 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return CompositeSignal(
            asset_key=asset_key,
            display_name=asset_cfg["display_name"],
            timestamp=ts,
            spot_price=spot_price,
            price_change_pct=price_change,
            sentiment=sent,
            technicals=tech,
            win_loss_ratio=win_loss_ratio,
            composite_score=composite,
            trade_signal=trade_signal,
            confidence=confidence,
            reasoning=reasoning,
        )

    # ══════════════════════════════════════════════════════════
    # UTILITY / DEBUG
    # ══════════════════════════════════════════════════════════

    def explain_divergence(self, signal: DivergenceSignal) -> str:
        """Human-readable explanation of a divergence signal."""
        explanations = {
            "bullish_regular": (
                "REGULAR BULLISH DIVERGENCE\n"
                "Price made a lower low while RSI made a higher low.\n"
                "This is a classic reversal signal indicating selling momentum is "
                "exhausting. Smart money is likely absorbing supply (accumulation).\n"
                "SMC: Look for a demand zone at this level for confluence."
            ),
            "bearish_regular": (
                "REGULAR BEARISH DIVERGENCE\n"
                "Price made a higher high while RSI made a lower high.\n"
                "Buying momentum is waning — a reversal to the downside is probable.\n"
                "SMC: Look for a supply zone at this level for confluence."
            ),
            "bullish_hidden": (
                "HIDDEN BULLISH DIVERGENCE\n"
                "Price made a higher low while RSI made a lower low.\n"
                "This is a trend continuation signal in an existing uptrend.\n"
                "Wyckoff: Consistent with a re-accumulation phase / spring."
            ),
            "bearish_hidden": (
                "HIDDEN BEARISH DIVERGENCE\n"
                "Price made a lower high while RSI made a higher high.\n"
                "Downtrend continuation expected.\n"
                "Wyckoff: Consistent with re-distribution before further markdown."
            ),
            "none": "No divergence pattern detected on this timeframe.",
        }
        base = explanations.get(signal.divergence_type, "Unknown signal.")
        return (
            f"{base}\n\n"
            f"Timeframe : {signal.timeframe}\n"
            f"Strength  : {signal.strength:.2%}\n"
            f"Price     : {signal.price_at_signal:.4f}\n"
            f"RSI       : {signal.rsi_at_signal:.2f}"
        )
