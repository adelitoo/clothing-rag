#!/bin/bash

echo "🔍 Checking Python virtual environment..."

if [ ! -d ".venv" ]; then
    echo "🛠️  Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "🐍 Activating virtual environment..."
source .venv/bin/activate

if ! pip show fastapi &> /dev/null; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies already installed."
fi

echo "🚀 Starting Milvus..."
bash milvus/standalone_embed.sh start

echo "🖥️  Starting FastAPI backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
