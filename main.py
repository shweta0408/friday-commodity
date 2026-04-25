"""
FRIDAY — main.py
================
Entry point. Can be run in two modes:

  1. Dashboard mode (default):
       python main.py
       → Launches Streamlit dashboard on http://localhost:8501

  2. CLI mode:
       python main.py --cli
       → Runs a single analysis cycle and prints Rich table to terminal

APScheduler handles background polling so the dashboard always
shows fresh data without blocking the UI thread.
"""

import argparse
import logging
import os
import sys
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FRIDAY.Main")


def run_streamlit():
    """Launch Streamlit dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    logger.info("Launching FRIDAY dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", "8501",
        "--server.headless", "false",
        "--theme.base", "dark",
        "--theme.backgroundColor", "#080B0F",
        "--theme.secondaryBackgroundColor", "#0D1117",
        "--theme.textColor", "#C8D0DC",
        "--theme.primaryColor", "#F5A623",
    ])


def run_cli():
    """Single-cycle CLI analysis using Rich for terminal output."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
    except ImportError:
        print("Rich not installed. Run: pip install rich")
        sys.exit(1)

    from data_streamer import DataStreamer
    from intelligence_unit import IntelligenceUnit
    from performance_tracker import PerformanceTracker
    from config import ASSETS

    console = Console()

    console.print(Panel.fit(
        "[bold yellow]⚡ FRIDAY COMMODITY INTELLIGENCE SYSTEM[/bold yellow]\n"
        "[dim]CLI Analysis Mode[/dim]",
        border_style="yellow",
    ))

    streamer = DataStreamer()
    intel    = IntelligenceUnit()
    tracker  = PerformanceTracker()

    table = Table(
        title=f"FRIDAY Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        box=box.MINIMAL_DOUBLE_HEAD,
        border_style="dim",
        show_header=True,
        header_style="bold dim yellow",
    )
    table.add_column("Asset",       style="bold white",  width=18)
    table.add_column("Price",       style="cyan",        width=12, justify="right")
    table.add_column("Chg%",        style="",            width=8,  justify="right")
    table.add_column("Sentiment",   style="",            width=10)
    table.add_column("15m Div",     style="",            width=20)
    table.add_column("Score",       style="bold",        width=8,  justify="right")
    table.add_column("Signal",      style="bold",        width=8)
    table.add_column("Confidence",  style="dim",         width=10)

    for asset_key in ASSETS:
        cfg = ASSETS[asset_key]
        console.print(f"[dim]Analysing {cfg['display_name']}...[/dim]")

        try:
            ohlcv_dict = streamer.get_all_ohlcv(asset_key)
            spot       = streamer.get_spot_price(asset_key) or 0.0
            change     = streamer.get_price_change_pct(asset_key) or 0.0
            news       = streamer.get_news(asset_key)
            win_rate   = tracker.asset_win_rate(asset_key)

            signal = intel.analyse(
                asset_key=asset_key,
                ohlcv_dict=ohlcv_dict,
                news_articles=news,
                win_loss_ratio=win_rate,
                spot_price=spot,
                price_change=change,
            )

            score = signal.composite_score
            score_color = "green" if score > 10 else "red" if score < -10 else "yellow"
            sig_color   = "green" if signal.trade_signal == "LONG" else \
                          "red"   if signal.trade_signal == "SHORT" else "yellow"
            chg_color   = "green" if change > 0 else "red"
            div15m      = signal.technicals.divergences.get("15m")
            div_txt     = div15m.divergence_type.replace("_"," ").title() if div15m else "None"
            div_color   = "green" if div15m and "bullish" in div15m.divergence_type else \
                          "red"   if div15m and "bearish" in div15m.divergence_type else "dim"

            table.add_row(
                f"{cfg['emoji']} {cfg['display_name']}",
                f"${spot:,.2f}",
                f"[{chg_color}]{change:+.2f}%[/{chg_color}]",
                signal.sentiment.tone,
                f"[{div_color}]{div_txt}[/{div_color}]",
                f"[{score_color}]{score:+.1f}[/{score_color}]",
                f"[{sig_color}]{signal.trade_signal}[/{sig_color}]",
                signal.confidence,
            )

            # Print reasoning
            console.print(f"\n[bold yellow]{cfg['emoji']} {cfg['display_name'].upper()} — Reasoning:[/bold yellow]")
            for reason in signal.reasoning:
                console.print(f"  [dim]{reason}[/dim]")

        except Exception as exc:
            logger.error(f"CLI error for {asset_key}: {exc}", exc_info=True)
            table.add_row(cfg["display_name"], "ERROR", "—", "—", "—", "—", "—", "—")

    console.print()
    console.print(table)

    # Performance summary
    summary = tracker.summary()
    console.print(Panel(
        f"Total Trades: {summary['total_trades']}  |  "
        f"Closed: {summary['closed_trades']}  |  "
        f"Global W/L: {summary['global_win_rate']:.1%}",
        title="[bold dim]SYSTEM PERFORMANCE[/bold dim]",
        border_style="dim",
    ))


def main():
    parser = argparse.ArgumentParser(description="FRIDAY Commodity Intelligence System")
    parser.add_argument("--cli", action="store_true", help="Run CLI analysis (no browser)")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_streamlit()


if __name__ == "__main__":
    main()
