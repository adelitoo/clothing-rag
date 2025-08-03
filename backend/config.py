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
LLM_JUDGE_MODEL = "qwen2.5:7b-instruct-q8_0"

# === Batch Sizes & Hardware ===
TEXT_BATCH_SIZE = 512
IMAGE_BATCH_SIZE = 64


def get_optimal_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA (NVIDIA GPU) detected.")
        return device
    elif torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T
            print("✅ Apple Silicon MPS detected and working.")
            return device
        except Exception as e:
            print(f"⚠️ MPS detected but not working properly: {e}")
            print("   Falling back to CPU for stability.")
            return torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("✅ No GPU detected, using CPU.")
        return device


DEVICE = get_optimal_device()

if DEVICE.type == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    NUM_WORKERS = 0
    PIN_MEMORY = False
elif DEVICE.type == "cuda":
    NUM_WORKERS = min(32, os.cpu_count() or 1)
    PIN_MEMORY = True
else:
    NUM_WORKERS = min(8, os.cpu_count() or 1)
    PIN_MEMORY = False

# === Milvus / Vector DB ===
EMB_DIM = 512
MILVUS_INSERT_BATCH_SIZE = 10000
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# === Redis ===
REDIS_HOST = "localhost"
REDIS_PORT = "6379"

# === Evaluation Settings ===
EVALUATION_K = 10
RELEVANCE_THRESHOLD = 2

# --- System Under Test Definitions ---
SYSTEMS_TO_EVALUATE = {
    "vector_search": {
        "name": "Vector Search (LLM)",
        "url": "http://127.0.0.1:8000/search/",
        "color": "#F44336",
    },
    "baseline": {
        "name": "Baseline",
        "url": "http://127.0.0.1:8000/search/baseline/",
        "color": "#4CAF50",
    },
}
