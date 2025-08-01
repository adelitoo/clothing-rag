#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "🔍 Checking Python virtual environment..."

if [ ! -d "$PROJECT_ROOT/backend/.venv" ]; then
    echo "🛠️  Creating virtual environment (.venv)..."
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

echo "🐍 Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

if ! pip show fastapi &> /dev/null; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/clothing-rag/backend/requirements.txt"
else
    echo "✅ Dependencies already installed."
fi

echo "🚀 Starting Milvus..."
bash "$PROJECT_ROOT/backend/milvus/standalone_embed.sh" start

echo "🖥️  Starting FastAPI backend..."
cd "$PROJECT_ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
