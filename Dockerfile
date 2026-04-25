# ============================================================
# FRIDAY — Commodity Intelligence System
# Dockerfile — uses Python 3.11 (stable, all wheels available)
# ============================================================

FROM python:3.11-slim

# System deps for scipy / numpy builds
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install faiss-cpu separately (pre-built wheel for py3.11)
RUN pip install --no-cache-dir faiss-cpu

# Install all other deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create data directory
RUN mkdir -p friday_data

EXPOSE 8501

# Streamlit config — headless for cloud
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_THEME_BASE=dark
ENV STREAMLIT_THEME_BACKGROUND_COLOR=#080B0F
ENV STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR=#0D1117
ENV STREAMLIT_THEME_TEXT_COLOR=#C8D0DC
ENV STREAMLIT_THEME_PRIMARY_COLOR=#F5A623

CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0"]
