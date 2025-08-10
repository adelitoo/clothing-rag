#!/bin/bash

echo "🚀 Starting Fashion Search Backend Environment..."

export TOKENIZERS_PARALLELISM=false
SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=$( cd -- "$( dirname -- "$SOURCE_PATH" )" &> /dev/null && pwd )
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

echo "🐍 Checking Python virtual environment..."
if [ ! -d ".venv" ]; then
    echo "   - Creating virtual environment (.venv)..."
    python3.11 -m venv ".venv"
fi
source ".venv/bin/activate"

echo "📦 Checking Python dependencies..."
if ! pip show fastapi &> /dev/null; then
    echo "   - Installing dependencies from requirements.txt..."
    pip install -r "requirements.txt"
else
    echo "   - Dependencies already installed."
fi

REDIS_CONTAINER_NAME="redis"
REDIS_IMAGE="redis/redis-stack-server:latest"

echo "🔍 Checking Redis container..."
if [ "$(docker ps -q -f name=$REDIS_CONTAINER_NAME)" ]; then
    echo "   - Redis container is already running."
elif [ "$(docker ps -aq -f status=exited -f name=$REDIS_CONTAINER_NAME)" ]; then
    echo "   - Starting existing Redis container..."
    docker start $REDIS_CONTAINER_NAME
else
    echo "   - Creating and starting new Redis container..."
    docker run -d --name $REDIS_CONTAINER_NAME -p 6379:6379 -p 8001:8001 $REDIS_IMAGE
    echo "   - Waiting for Redis to initialize..."
    sleep 5
fi

echo "💾 Checking for data in Redis..."
KEY_COUNT=$(docker exec $REDIS_CONTAINER_NAME redis-cli DBSIZE)

if [ "$KEY_COUNT" -eq 0 ]; then
    echo "   - Redis is empty. Loading data from CSV..."
    export PYTHONPATH="$PROJECT_ROOT"
    python -m scripts.load_redis_data
else
    echo "   - Redis already contains $KEY_COUNT keys. Skipping data load."
fi

echo "🧠 Starting Milvus..."
bash "$PROJECT_ROOT/milvus/standalone_embed.sh" start

echo "🖥️  Starting FastAPI backend on http://0.0.0.0:8000"
uvicorn src.fashion_search.api.main:app --host 0.0.0.0 --port 8000 --reload --app-dir .
