# 🧵 Fashion Search Backend

This is a FastAPI-based backend for fashion-related image and text search using CLIP embeddings, Milvus vector database, and query rewriting via Ollama's LLM.

---

## 🚀 Getting Started

You have **two ways** to run the backend:

### ✅ Recommended: Use the `start_backend.sh` Bash Script

From the project root:

1. **Give execution permissions once**:
   ```bash
   chmod +x scripts/start_backend.sh
   ```

2. **First time only** – run with `source` to activate the `.venv`:
   ```bash
   source scripts/start_backend.sh
   ```

3. **After that**, just use:
   ```bash
   .scripts/start_backend.sh
   ```

> This script automates: **dependencies**, **starting Milvus**, **activating the Python environment**, and **launching the FastAPI backend**.  
> You still need to manually download the Ollama LLM model during first-time setup.

---

## 🧠 Step-by-Step Manual Setup

### 1. ✅ Setup Ollama with the LLM

Download and install Ollama from: [https://ollama.com](https://ollama.com)

Start the Ollama server (if not started automatically):
```bash
ollama serve
```

Pull the required model:
```bash
ollama pull llama3.1:8b
```

---

### 2. 🧱 Start Milvus Vector Database

From the project root:

```bash
bash milvus/standalone_embed.sh start
```

This will spin up the Milvus container for vector similarity search.

---

### 3. 🐍 Set up the Python Environment

#### Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 4. 🖥️ Start the FastAPI Server

Back in the project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser:
- API Root: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

---

## ✅ Summary

- **Ollama** transforms fashion queries using a local LLM.
- **Milvus** stores and searches CLIP text/image embeddings.
- **FastAPI** provides the search endpoint.
- You can run everything manually or just use the provided script.

---
