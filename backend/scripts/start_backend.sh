#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=$( cd -- "$( dirname -- "$SOURCE_PATH" )" &> /dev/null && pwd )
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🔍 Checking Python virtual environment..."
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "🛠️  Creating virtual environment (.venv)..."
    python3.11 -m venv "$PROJECT_ROOT/.venv"
fi

echo "🐍 Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

if ! pip show fastapi &> /dev/null; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "✅ Dependencies already installed."
fi

REDIS_CONTAINER_NAME="redis"
REDIS_IMAGE="redis/redis-stack-server:latest"

echo "🔍 Checking Redis container..."

if [ "$(docker ps -q -f name=$REDIS_CONTAINER_NAME)" ]; then
    echo "✅ Redis container is already running."
elif [ "$(docker ps -aq -f status=exited -f name=$REDIS_CONTAINER_NAME)" ]; then
    echo "🔄 Starting existing Redis container..."
    docker start $REDIS_CONTAINER_NAME
else
    echo "🚀 Creating and starting Redis Stack container..."
    docker run -d --name $REDIS_CONTAINER_NAME -p 6379:6379 -p 8001:8001 $REDIS_IMAGE
fi

echo "🚀 Starting Milvus..."
bash "$PROJECT_ROOT/milvus/standalone_embed.sh" start

echo "🖥️  Starting FastAPI backend..."
cd "$PROJECT_ROOT"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
