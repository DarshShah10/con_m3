#!/bin/bash
set -e

echo "=============================="
echo "🚀 Bootstrapping environment"
echo "=============================="

# -------- System dependencies --------
echo "🔧 Updating apt..."
apt-get update -y

echo "🎥 Installing system packages..."
apt-get install -y \
    ffmpeg \
    sox \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    libsndfile1 \
    ca-certificates

# -------- Python virtual environment --------
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python venv (venv)..."
    python3 -m venv venv
else
    echo "🐍 venv already exists, skipping creation"
fi

echo "📌 Activating venv..."
source venv/bin/activate

echo "⬆️ Upgrading pip tooling..."
pip install --upgrade pip setuptools wheel

# -------- Python dependencies --------
if [ ! -f "requirements.lock" ]; then
    echo "❌ ERROR: requirements.lock not found!"
    exit 1
fi

echo "📦 Installing Python dependencies from requirements.lock..."
pip install --no-cache-dir -r requirements.lock

echo "=============================="
echo "✅ Environment ready"
echo "=============================="

echo ""
echo "To activate later:"
echo "  source venv/bin/activate"
