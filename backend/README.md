# 🧵 Fashion Search Backend

This project provides a powerful, multi-modal fashion search API. It leverages CLIP embeddings for understanding image and text content, a Milvus vector database for high-speed similarity search, Redis for caching, and a local LLM (via Ollama) to enhance user queries for more intuitive results.

## ✨ Features

- **Vector Search**: Finds fashion items based on descriptive text queries (e.g., "blue jeans for men")
- **LLM Query Enhancement**: Automatically rewrites simple queries into detailed descriptions to improve search relevance
- **High Performance**: Uses Milvus for fast vector indexing and Redis for caching frequent search results
- **Modern API**: Built with FastAPI, providing interactive documentation via Swagger UI

## 🛠️ Installation

Follow these steps to set up the project on a macOS or Linux machine.

### 1. Set Up the Python Environment

It's highly recommended to use a stable version of Python, like 3.11. Tools like `pyenv` can help manage Python versions.

```bash
# Navigate to the project directory
cd /path/to/your/project

# Create a virtual environment using Python 3.11
python3.11 -m venv .venv

# Activate the environment
source .venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### 3. Set Up the LLM (Ollama)

This project uses a local LLM for query enhancement.

```bash
# Install Ollama by following instructions at https://ollama.com

# Pull the required model
ollama pull llama3.1:8b
```

## 🚀 Usage

A single script handles starting all the necessary services (Redis, Milvus, and the FastAPI server).

1. **Make the script executable** (only needs to be done once):

```bash
chmod +x scripts/start_backend.sh
```

2. **Run the script**:

```bash
./scripts/start_backend.sh
```

This script will:
- Activate the Python virtual environment
- Start the Redis and Milvus Docker containers
- Launch the FastAPI application

Your backend is now running and available at `http://0.0.0.0:8000`.

## 📡 API Endpoints

Once the server is running, you can access the interactive API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

The primary search endpoint is:
- **POST /search/**: Submits a search query and returns the most relevant fashion items

## 💻 Technology Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Milvus
- **Caching**: Redis (via Redis Stack for RedisJSON)
- **LLM Provider**: Ollama
- **Containerization**: Docker
- **Embedding Models**: CLIP (for fashion embeddings)

## 🔧 Manual Setup (Alternative)

If you prefer to set up services manually instead of using the automated script:

### Start Redis
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### Start Milvus
```bash
bash milvus/standalone_embed.sh start
```

### Start Ollama Server
```bash
ollama serve
```

### Start FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📝 Notes

- Ensure Docker is installed and running on your system
- The first run may take longer as Docker images are downloaded
- Redis caching significantly improves response times for repeated queries
- The LLM query enhancement can be toggled on/off via API parameters

---

