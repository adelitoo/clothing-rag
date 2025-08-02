# 🧵 Fashion Search Project 

A full-stack fashion search application that uses CLIP embeddings, vector similarity search, and LLM-powered query enhancement to find visually similar fashion items.

## 🏗️ Architecture

This project consists of two main components:

- **Backend** (`/backend/`): FastAPI server with CLIP embeddings, Milvus vector database, and Ollama LLM integration
- **Frontend** (`/frontend/`): Streamlit web interface for search queries and results visualization

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Milvus)
- Git

### 1. Clone the Repository

```bash
git clone git@github.com:adelitoo/clothing-rag.git
cd clothing-rag
```

### 2. Backend Setup

#### Option A: Automated Setup (Recommended)

Navigate to the backend directory and use the provided script:

```bash
cd backend
chmod +x scripts/start_backend.sh

# First time - run with source to activate environment
source scripts/start_backend.sh

# Subsequent runs
./scripts/start_backend.sh
```

#### Option B: Manual Setup

**Install Ollama LLM:**
1. Download from [https://ollama.com](https://ollama.com)
2. Start the server: `ollama serve`
3. Pull the model: `ollama pull llama3.1:8b`

**Start Milvus Vector Database:**
```bash
cd backend
bash milvus/standalone_embed.sh start
```

**Setup Python Environment:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**Start FastAPI Server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup

**Setup Python Environment:**
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**Insert Embeddings into Milvus:**
```bash
python send_request.py --mode pipeline --insert-db
```

**Launch Streamlit App:**
```bash
streamlit run app.py
```

---

## 🌐 Access Points

Once everything is running:

- **Backend API**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Frontend App**: `http://localhost:8501`

---

## 📁 Project Structure

```
fashion-search-project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Backend dependencies
│   ├── scripts/
│   │   └── start_backend.sh # Automated setup script
│   └── milvus/
│       └── standalone_embed.sh
├── frontend/
│   ├── app.py              # Streamlit application
│   ├── send_request.py     # Embedding insertion script
│   └── requirements.txt    # Frontend dependencies
└── README.md               # This file
```

---

## 🔧 How It Works

1. **Query Processing**: User submits a fashion-related text query through the Streamlit frontend
2. **LLM Enhancement**: Ollama's LLM refines and expands the query for better search results
3. **Embedding Generation**: CLIP model generates embeddings for the enhanced query
4. **Vector Search**: Milvus performs similarity search against stored text embeddings
5. **Results Display**: Frontend displays visually similar fashion items

---

## 🛠️ Technologies Used

- **FastAPI**: High-performance web framework for the backend API
- **Streamlit**: Frontend web application framework
- **Milvus**: Vector database for similarity search
- **Ollama**: Local LLM for query enhancement
- **CLIP**: Vision-language model for embeddings
- **BLIP**: Image labelling model
- **Docker**: Containerization for Milvus

---

## 📝 Development Notes

### Adding New Data

To add new fashion items to the database:
1. Place images in the appropriate directory
2. Run the embedding pipeline: `python frontend/send_request.py --mode pipeline --cleanup --caption --embed --insert-db`

### Customizing the LLM Model

To use a different Ollama model:
1. Pull the desired model: `ollama pull <model-name>`
2. Update the model reference in the backend configuration

### Environment Variables

Create `.env` files in both `backend/` and `frontend/` directories if you need to customize:
- Database connection strings
- API endpoints
- Model configurations

---

## 🔍 Troubleshooting

### Common Issues

**Milvus Connection Error:**
- Ensure Docker is running
- Check if Milvus container is started: `docker ps`
- Restart Milvus: `bash backend/milvus/standalone_embed.sh restart`

**Ollama Model Not Found:**
- Verify Ollama is running: `ollama list`
- Pull the required model: `ollama pull llama3.1:8b`

**Port Conflicts:**
- Backend runs on port 8000
- Frontend runs on port 8501
- Milvus uses port 19530
- Ensure these ports are available

**Python Environment Issues:**
- Make sure virtual environments are activated
- Reinstall dependencies if needed: `pip install -r requirements.txt`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

---

## 🙋‍♂️ Support

For questions or issues:
- Check the troubleshooting section above
- Open an issue in the repository
- Review the API documentation at `http://localhost:8000/docs`
