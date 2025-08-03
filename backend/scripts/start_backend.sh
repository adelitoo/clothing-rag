#!/bin/bash

# --- Setup and Path Definition ---
SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=$( cd -- "$( dirname -- "$SOURCE_PATH" )" &> /dev/null && pwd )
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# --- Python Virtual Environment ---
echo "🔍 Checking Python virtual environment..."
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "🛠️  Creating virtual environment (.venv)..."
    python3.11 -m venv "$PROJECT_ROOT/.venv"
fi

echo "🐍 Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# --- Python Dependencies ---
if ! pip show fastapi &> /dev/null; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "✅ Dependencies already installed."
fi

# --- Redis Container ---
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
    sleep 5
fi

# --- Check and Load Redis Data ---
echo "🔍 Checking for data in Redis..."
KEY_COUNT=$(docker exec $REDIS_CONTAINER_NAME redis-cli DBSIZE)

if [ "$KEY_COUNT" -eq 0 ]; then
    echo " Redis is empty. Loading data from CSV..."
    
    # --- THIS IS THE FIX ---
    # Set PYTHONPATH to the project root so Python can find the 'src' module.
    # Then run the script as a module (-m) which is more robust.
    export PYTHONPATH="$PROJECT_ROOT"
    python -m scripts.load_redis_data
    # ----------------------

else
    echo "✅ Redis already contains $KEY_COUNT keys. Skipping data load."
fi

# --- Milvus Vector Database ---
echo "🚀 Starting Milvus..."
bash "$PROJECT_ROOT/milvus/standalone_embed.sh" start

# --- FastAPI Backend ---
echo "🖥️  Starting FastAPI backend..."
cd "$PROJECT_ROOT"
uvicorn src.fashion_search.api.main:app --host 0.0.0.0 --port 8000 --reload --app-dir .
