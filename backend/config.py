import os
import torch
from pathlib import Path

# === Project Root ===
BASE_DIR = Path(__file__).parent.resolve()

# === Core Paths ===
DATA_DIR = BASE_DIR / "data"
EVALUATION_DIR = BASE_DIR / "evaluation"
IMAGE_BASE_DIR = DATA_DIR / "images"
REPORTS_DIR = EVALUATION_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"

# --- Data Files ---
ARTICLES_CSV_PATH = DATA_DIR / "articles.csv"
COMPLETE_ARTICLES_CSV_PATH = DATA_DIR / "complete_articles.csv"
EMBEDDING_SAVE_PATH = DATA_DIR / "embeddings.npz"
QUERIES_FILE_PATH = DATA_DIR / "fashion_queries.csv"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.csv" 
ANNOTATION_FILE_OUTPUT = REPORTS_DIR / "to_annotate.csv"

# === Models ===
IMAGE_TEXT_MODEL = "patrickjohncyh/fashion-clip"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
LLM_JUDGE_MODEL = "llama3.1:8b"

# === Batch Sizes & Hardware ===
TEXT_BATCH_SIZE = 512
IMAGE_BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(32, os.cpu_count() or 1)
PIN_MEMORY = DEVICE.type == "cuda"

# === Milvus / Vector DB ===
EMB_DIM = 512
MILVUS_INSERT_BATCH_SIZE = 10000
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# === Evaluation Settings ===
EVALUATION_K = 10  
RELEVANCE_THRESHOLD = 2 

# --- System Under Test Definitions ---
SYSTEMS_TO_EVALUATE = {
    "vector_search": {
        "name": "Vector Search (LLM)",
        "url": "http://127.0.0.1:8000/search/",
        "color": "#F44336" 
    },
    "baseline": {
        "name": "Baseline",
        "url": "http://127.0.0.1:8000/search/baseline/",
        "color": "#4CAF50" 
    }
}