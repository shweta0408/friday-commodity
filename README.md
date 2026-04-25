---
title: FRIDAY Commodity Intelligence
emoji: 🛢️
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.35.0
app_file: dashboard.py
pinned: false
python_version: "3.11"
---

# FRIDAY — Commodity Intelligence System

A real-time commodity market analysis dashboard covering **Crude Oil, Gold, and Silver** using:

- 📊 Multi-timeframe Technical Analysis (RSI Divergence, MACD, Bollinger Bands)
- 🧠 NLP Sentiment Analysis (VADER + Sentence Transformers)
- 📰 News-driven Signal Generation
- 🏆 Performance Tracking (Win/Loss ratio)

## Setup — Hugging Face Spaces

Set the following **Secrets** in your Space settings (Settings → Variables and secrets):

| Secret | Description |
|--------|-------------|
| `FINNHUB_API_KEY` | Finnhub.io API key (free tier works) |
| `NEWSAPI_KEY` | NewsAPI key (optional) |
| `ALPHA_VANTAGE_KEY` | Alpha Vantage key (optional) |

## Running Locally

```bash
cp .env.template .env
# Fill in your API keys in .env
pip install faiss-cpu  # install separately first
pip install -r requirements.txt
streamlit run dashboard.py
```
