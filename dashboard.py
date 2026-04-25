"""
FRIDAY — dashboard.py
======================
class FridayDashboard
Streamlit-based live terminal dashboard for the FRIDAY system.

Design philosophy:
  • Dark military/HFT aesthetic — deep blacks, amber accents, monospace type
  • Real-time auto-refresh via streamlit-autorefresh
  • One card per asset with all composite signal data
  • Reasoning panel, divergence table, trade log
  • APScheduler handles background data polling
"""

import time
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import ASSETS, TIMEFRAMES, DATA_DIR, TRADE_LOG_PATH
from data_streamer import DataStreamer
from intelligence_unit import IntelligenceUnit, CompositeSignal
from performance_tracker import PerformanceTracker

logger = logging.getLogger("FRIDAY.Dashboard")

# ── Page Config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="FRIDAY | Commodity Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — FRIDAY DARK THEME
# ═══════════════════════════════════════════════════════════════

FRIDAY_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700;900&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #080B0F !important;
    color: #C8D0DC !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1rem 2rem !important; max-width: 100% !important; }

  /* ── Header Bar ── */
  .friday-header {
    background: linear-gradient(90deg, #0A0D12 0%, #0F1620 50%, #0A0D12 100%);
    border-bottom: 1px solid #F5A623;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
  }
  .friday-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 900;
    color: #F5A623;
    letter-spacing: 6px;
    text-transform: uppercase;
  }
  .friday-subtitle {
    font-size: 11px;
    color: #506080;
    letter-spacing: 3px;
  }
  .friday-clock {
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    color: #F5A623;
    opacity: 0.8;
  }

  /* ── Asset Cards ── */
  .asset-card {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
  }
  .asset-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .card-long::before    { background: #00E676; }
  .card-short::before   { background: #FF3D57; }
  .card-hold::before    { background: #F5A623; }

  /* ── Score Meter ── */
  .score-meter {
    font-family: 'Orbitron', sans-serif;
    font-size: 40px;
    font-weight: 900;
    line-height: 1;
  }
  .score-positive { color: #00E676; }
  .score-negative { color: #FF3D57; }
  .score-neutral  { color: #F5A623; }

  /* ── Signal Badge ── */
  .signal-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 2px;
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
  }
  .badge-long  { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid #00E676; }
  .badge-short { background: rgba(255,61,87,0.15);  color: #FF3D57; border: 1px solid #FF3D57; }
  .badge-hold  { background: rgba(245,166,35,0.10); color: #F5A623; border: 1px solid #F5A623; }

  /* ── Confidence ── */
  .conf-high   { color: #00E676; font-size: 11px; letter-spacing: 2px; }
  .conf-medium { color: #F5A623; font-size: 11px; letter-spacing: 2px; }
  .conf-low    { color: #8899AA; font-size: 11px; letter-spacing: 2px; }

  /* ── Data Table ── */
  .data-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #1A2232;
    font-size: 12px;
  }
  .data-label { color: #506080; letter-spacing: 1px; }
  .data-value { color: #C8D0DC; font-weight: 700; }

  /* ── Reasoning Panel ── */
  .reasoning-item {
    padding: 6px 10px;
    margin: 4px 0;
    background: #0A0D12;
    border-left: 2px solid #2A3A50;
    font-size: 11px;
    color: #8899AA;
    line-height: 1.5;
  }

  /* ── Streamlit metric override ── */
  [data-testid="metric-container"] {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 4px;
    padding: 12px;
  }
  [data-testid="metric-container"] label { color: #506080 !important; font-size: 11px !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F5A623 !important;
    font-family: 'Orbitron', sans-serif !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #080B0F; }
  ::-webkit-scrollbar-thumb { background: #2A3A50; border-radius: 2px; }

  /* ── Divider ── */
  hr { border-color: #1E2A3A !important; margin: 12px 0; }

  /* ── Section headers ── */
  .section-header {
    font-size: 10px;
    letter-spacing: 4px;
    color: #506080;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #1E2A3A;
    padding-bottom: 4px;
  }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# DASHBOARD CLASS
# ═══════════════════════════════════════════════════════════════

class FridayDashboard:
    """Streamlit UI engine for the FRIDAY system."""

    def __init__(self):
        self.streamer = DataStreamer()
        self.intel    = IntelligenceUnit()
        self.tracker  = PerformanceTracker()
        self._signals: Dict[str, CompositeSignal] = {}

    # ── Data Refresh ─────────────────────────────────────────

    def _refresh_all(self):
        """Fetch data and run analysis for all assets."""
        for asset_key in ASSETS:
            try:
                ohlcv_dict  = self.streamer.get_all_ohlcv(asset_key)
                spot        = self.streamer.get_spot_price(asset_key) or 0.0
                change      = self.streamer.get_price_change_pct(asset_key) or 0.0
                news        = self.streamer.get_news(asset_key)
                win_rate    = self.tracker.asset_win_rate(asset_key)

                signal = self.intel.analyse(
                    asset_key=asset_key,
                    ohlcv_dict=ohlcv_dict,
                    news_articles=news,
                    win_loss_ratio=win_rate,
                    spot_price=spot,
                    price_change=change,
                )
                self._signals[asset_key] = signal

            except Exception as exc:
                logger.error(f"Error refreshing {asset_key}: {exc}", exc_info=True)

    # ── HTML Components ──────────────────────────────────────

    @staticmethod
    def _score_color_class(score: float) -> str:
        if score > 10:  return "score-positive"
        if score < -10: return "score-negative"
        return "score-neutral"

    @staticmethod
    def _badge_class(signal: str) -> str:
        return {"LONG": "badge-long", "SHORT": "badge-short"}.get(signal, "badge-hold")

    @staticmethod
    def _card_class(signal: str) -> str:
        return {"LONG": "card-long", "SHORT": "card-short"}.get(signal, "card-hold")

    @staticmethod
    def _conf_class(conf: str) -> str:
        return {"HIGH": "conf-high", "MEDIUM": "conf-medium"}.get(conf, "conf-low")

    @staticmethod
    def _format_price(price: float, asset_key: str) -> str:
        decimals = 3 if asset_key == "CRUDE_OIL" else 2
        return f"${price:,.{decimals}f}"

    # ── Plotly Charts ────────────────────────────────────────

    def _render_mini_chart(self, asset_key: str) -> Optional[go.Figure]:
        """Render a compact candlestick + RSI chart for the asset."""
        try:
            df = self.streamer.get_ohlcv(asset_key, "1h")
            if df is None or df.empty or len(df) < 20:
                return None
            df = df.tail(48)  # Last 48 hours of 1h candles

            # Calculate RSI
            from intelligence_unit import IntelligenceUnit
            rsi = IntelligenceUnit._calc_rsi(df["Close"].astype(float))

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                increasing_line_color="#00E676",
                decreasing_line_color="#FF3D57",
                increasing_fillcolor="rgba(0,230,118,0.4)",
                decreasing_fillcolor="rgba(255,61,87,0.4)",
                name="Price", showlegend=False,
            ))

            fig.update_layout(
                paper_bgcolor="#080B0F",
                plot_bgcolor="#0D1117",
                font=dict(family="JetBrains Mono", color="#506080", size=9),
                margin=dict(l=4, r=4, t=4, b=4),
                height=140,
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False,
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#1A2232", zeroline=False,
                    tickfont=dict(size=8),
                ),
            )
            return fig
        except Exception:
            return None

    def _render_score_gauge(self, score: float) -> go.Figure:
        """Semi-circular gauge for composite score."""
        color = "#00E676" if score > 10 else "#FF3D57" if score < -10 else "#F5A623"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"color": color, "family": "Orbitron", "size": 24}, "suffix": ""},
            gauge={
                "axis":     {"range": [-100, 100], "tickfont": {"size": 8, "color": "#506080"}},
                "bar":      {"color": color, "thickness": 0.3},
                "bgcolor":  "#0D1117",
                "bordercolor": "#1E2A3A",
                "steps": [
                    {"range": [-100, -75], "color": "rgba(255,61,87,0.2)"},
                    {"range": [-75, -25],  "color": "rgba(255,61,87,0.07)"},
                    {"range": [-25, 25],   "color": "rgba(245,166,35,0.05)"},
                    {"range": [25, 75],    "color": "rgba(0,230,118,0.07)"},
                    {"range": [75, 100],   "color": "rgba(0,230,118,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#F5A623", "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=160,
            font=dict(family="JetBrains Mono", color="#506080"),
        )
        return fig

    # ── Asset Card ───────────────────────────────────────────

    def _render_asset_card(self, signal: CompositeSignal):
        asset_key  = signal.asset_key
        cfg        = ASSETS[asset_key]
        score      = signal.composite_score
        card_cls   = self._card_class(signal.trade_signal)
        badge_cls  = self._badge_class(signal.trade_signal)
        score_cls  = self._score_color_class(score)
        conf_cls   = self._conf_class(signal.confidence)

        price_str  = self._format_price(signal.spot_price, asset_key)
        change_str = f"{signal.price_change_pct:+.2f}%" if signal.price_change_pct else "—"
        change_clr = "color:#00E676" if signal.price_change_pct and signal.price_change_pct > 0 else "color:#FF3D57"

        st.markdown(f"""
        <div class="asset-card {card_cls}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div style="font-size:11px;color:#506080;letter-spacing:3px">{cfg['emoji']} {cfg['unit']}</div>
              <div style="font-family:'Orbitron',sans-serif;font-size:18px;color:#C8D0DC;font-weight:700;letter-spacing:2px">
                {cfg['display_name'].upper()}
              </div>
              <div style="font-size:28px;font-weight:700;color:#C8D0DC;margin:4px 0">
                {price_str}
                <span style="font-size:14px;{change_clr};margin-left:8px">{change_str}</span>
              </div>
            </div>
            <div style="text-align:right">
              <div class="score-meter {score_cls}">{score:+.0f}</div>
              <div style="font-size:9px;color:#506080;letter-spacing:2px">COMPOSITE SCORE</div>
              <div style="margin-top:8px">
                <span class="signal-badge {badge_cls}">{signal.trade_signal}</span>
              </div>
              <div class="{conf_cls}" style="margin-top:4px">▲ {signal.confidence} CONFIDENCE</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed Signal Panel ────────────────────────────────

    def _render_signal_detail(self, signal: CompositeSignal):
        tech = signal.technicals
        sent = signal.sentiment

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.markdown('<div class="section-header">TECHNICALS</div>', unsafe_allow_html=True)
            for tf in ["15m", "1h", "1d"]:
                rsi_val = tech.rsi_values.get(tf, "—")
                macd    = tech.macd_signals.get(tf, "—")
                div     = tech.divergences.get(tf)
                div_txt = div.divergence_type.replace("_", " ").upper() if div else "NONE"
                div_clr = "#00E676" if div and "bullish" in div.divergence_type else \
                          "#FF3D57" if div and "bearish" in div.divergence_type else "#506080"

                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{tf} RSI</span>
                  <span class="data-value">{rsi_val:.1f if isinstance(rsi_val, float) else rsi_val}</span>
                </div>
                <div class="data-row">
                  <span class="data-label">{tf} MACD</span>
                  <span class="data-value">{macd.upper()}</span>
                </div>
                <div class="data-row">
                  <span class="data-label">{tf} DIV</span>
                  <span style="color:{div_clr};font-size:10px;font-weight:700">{div_txt}</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">S/D ZONES</div>', unsafe_allow_html=True)
            for lo, hi in tech.supply_zones[-3:]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label" style="color:#FF3D57">⬤ SUPPLY</span>
                  <span class="data-value">{lo:.2f}–{hi:.2f}</span>
                </div>""", unsafe_allow_html=True)
            for lo, hi in tech.demand_zones[-3:]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label" style="color:#00E676">⬤ DEMAND</span>
                  <span class="data-value">{lo:.2f}–{hi:.2f}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:12px">KEY LEVELS</div>', unsafe_allow_html=True)
            for r in tech.resistance_levels[:3]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label" style="color:#FF3D57">R</span>
                  <span class="data-value">{r:.2f}</span>
                </div>""", unsafe_allow_html=True)
            for s in tech.support_levels[:3]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label" style="color:#00E676">S</span>
                  <span class="data-value">{s:.2f}</span>
                </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="section-header">SENTIMENT</div>', unsafe_allow_html=True)
            tone_clr = "#00E676" if sent.tone == "Bullish" else "#FF3D57" if sent.tone == "Bearish" else "#F5A623"
            st.markdown(f"""
            <div class="data-row"><span class="data-label">TONE</span>
              <span style="color:{tone_clr};font-weight:700">{sent.tone.upper()}</span></div>
            <div class="data-row"><span class="data-label">VADER</span>
              <span class="data-value">{sent.vader_score:+.3f}</span></div>
            <div class="data-row"><span class="data-label">SCORE</span>
              <span class="data-value">{sent.sentiment_score:+.3f}</span></div>
            <div class="data-row"><span class="data-label">ARTICLES</span>
              <span class="data-value">{sent.article_count}</span></div>
            """, unsafe_allow_html=True)

            if sent.impact_matches:
                st.markdown('<div class="section-header" style="margin-top:12px">IMPACT EVENTS</div>', unsafe_allow_html=True)
                for m in sent.impact_matches[:2]:
                    move_clr = "#00E676" if m["move_pct"] > 0 else "#FF3D57"
                    st.markdown(f"""
                    <div style="padding:6px;background:#0A0D12;border-left:2px solid {move_clr};margin:4px 0;font-size:10px">
                      <div style="color:#8899AA">{m['historical_headline'][:55]}...</div>
                      <div style="margin-top:2px">
                        <span style="color:{move_clr};font-weight:700">{m['move_pct']:+.1f}%</span>
                        <span style="color:#506080;margin-left:8px">sim={m['similarity']*100:.0f}%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="section-header">REASONING</div>', unsafe_allow_html=True)
            for line in signal.reasoning:
                st.markdown(f'<div class="reasoning-item">{line}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:12px">WYCKOFF</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="padding:8px;background:#0A0D12;border:1px solid #1E2A3A;font-size:11px;color:#F5A623">
              {tech.wyckoff_phase}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:12px">W/L RATIO</div>', unsafe_allow_html=True)
            wl = signal.win_loss_ratio
            wl_clr = "#00E676" if wl >= 0.55 else "#FF3D57" if wl < 0.45 else "#F5A623"
            st.markdown(f"""
            <div style="font-family:'Orbitron',sans-serif;font-size:22px;color:{wl_clr};font-weight:700">
              {wl:.1%}
            </div>""", unsafe_allow_html=True)

    # ── Main Summary Table ───────────────────────────────────

    def _render_summary_table(self):
        if not self._signals:
            return

        rows = []
        for key, sig in self._signals.items():
            cfg = ASSETS[key]
            rows.append({
                "Asset":       f"{cfg['emoji']} {cfg['display_name']}",
                "Price":       self._format_price(sig.spot_price, key),
                "Chg %":       f"{sig.price_change_pct:+.2f}%" if sig.price_change_pct else "—",
                "Sentiment":   sig.sentiment.tone,
                "15m Div":     sig.technicals.divergences.get("15m", None) and
                               sig.technicals.divergences["15m"].divergence_type.replace("_"," ").title() or "None",
                "1h MACD":     sig.technicals.macd_signals.get("1h", "—").replace("_"," ").title(),
                "Score":       f"{sig.composite_score:+.1f}",
                "Signal":      sig.trade_signal,
                "Confidence":  sig.confidence,
                "W/L":         f"{sig.win_loss_ratio:.1%}",
            })

        df = pd.DataFrame(rows)

        def color_signal(val):
            if val == "LONG":   return "color: #00E676; font-weight: bold"
            if val == "SHORT":  return "color: #FF3D57; font-weight: bold"
            return "color: #F5A623"

        def color_score(val):
            try:
                v = float(val)
                if v > 10:  return "color: #00E676"
                if v < -10: return "color: #FF3D57"
            except:
                pass
            return "color: #F5A623"

        styled = df.style \
            .applymap(color_signal, subset=["Signal"]) \
            .applymap(color_score,  subset=["Score"]) \
            .set_properties(**{
                "background-color": "#0D1117",
                "color": "#C8D0DC",
                "border": "1px solid #1E2A3A",
                "font-size": "12px",
                "font-family": "JetBrains Mono",
            }) \
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", "#0A0D12"),
                    ("color", "#506080"),
                    ("font-size", "10px"),
                    ("letter-spacing", "2px"),
                    ("border", "1px solid #1E2A3A"),
                ]
            }])

        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Main Render Loop ─────────────────────────────────────

    def run(self):
        """Entry point — render the full FRIDAY dashboard."""
        # ── Auto-refresh every 60 seconds ─────────────────
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key="friday_refresh")
        except ImportError:
            pass

        # ── Inject CSS ────────────────────────────────────
        st.markdown(FRIDAY_CSS, unsafe_allow_html=True)

        # ── Header ────────────────────────────────────────
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
        st.markdown(f"""
        <div class="friday-header">
          <div>
            <div class="friday-title">⚡ FRIDAY</div>
            <div class="friday-subtitle">COMMODITY INTELLIGENCE SYSTEM  •  CRUDE OIL · GOLD · SILVER</div>
          </div>
          <div class="friday-clock">{now_str}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Refresh Controls ──────────────────────────────
        col_ref, col_status = st.columns([1, 5])
        with col_ref:
            if st.button("⟳ REFRESH", key="manual_refresh"):
                with st.spinner("Fetching live data..."):
                    self._refresh_all()
                st.success("Updated.")

        # ── Initial load / cached ─────────────────────────
        if not self._signals:
            with st.spinner("🛰️ FRIDAY initialising — fetching live data..."):
                self._refresh_all()

        if not self._signals:
            st.error("No data available. Check your API keys in .env and internet connectivity.")
            return

        # ── Summary Table ─────────────────────────────────
        st.markdown('<div class="section-header">LIVE MARKET OVERVIEW</div>', unsafe_allow_html=True)
        self._render_summary_table()

        # ── Asset Cards ───────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:24px">ASSET INTELLIGENCE</div>',
                    unsafe_allow_html=True)

        for asset_key, signal in self._signals.items():
            self._render_asset_card(signal)

            with st.expander(f"▸ EXPAND: {ASSETS[asset_key]['display_name'].upper()} FULL ANALYSIS", expanded=False):
                # Gauge + mini chart side by side
                gc, cc = st.columns([1, 2])
                with gc:
                    st.plotly_chart(
                        self._render_score_gauge(signal.composite_score),
                        use_container_width=True, key=f"gauge_{asset_key}"
                    )
                with cc:
                    chart = self._render_mini_chart(asset_key)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"chart_{asset_key}")
                    else:
                        st.caption("Insufficient data for chart.")

                self._render_signal_detail(signal)

                # Top headline
                if signal.sentiment.top_headline:
                    st.markdown(f"""
                    <div style="padding:10px;background:#0A0D12;border-left:3px solid #F5A623;
                                margin-top:8px;font-size:11px;color:#8899AA">
                      📰 {signal.sentiment.top_headline}
                    </div>""", unsafe_allow_html=True)

        # ── Performance Panel ─────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:24px">SYSTEM PERFORMANCE</div>',
                    unsafe_allow_html=True)
        summary = self.tracker.summary()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades",  summary["total_trades"])
        m2.metric("Closed Trades", summary["closed_trades"])
        m3.metric("Open Trades",   summary["open_trades"])
        m4.metric("Global W/L",    f"{summary['global_win_rate']:.1%}")
        m5.metric("Signals Today", len(self._signals))

        # ── Footer ────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:24px 0 8px;color:#2A3A50;font-size:10px;letter-spacing:3px">
          FRIDAY INTELLIGENCE SYSTEM  •  FOR INFORMATIONAL PURPOSES ONLY  •  NOT FINANCIAL ADVICE
        </div>""", unsafe_allow_html=True)


# ── Streamlit entry point ────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    dashboard = FridayDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
