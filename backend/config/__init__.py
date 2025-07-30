import os
import torch

# === Paths ===
CSV_PATH = "data/articles.csv"
IMAGE_BASE_DIR = "data/images"
EMBEDDING_SAVE_PATH = "data/embeddings.npz"
COMPLETE_CSV_PATH = "data/complete_articles.csv"
FILTERED_CSV_PATH = "data/articles.csv"
QUERIES_FILE_PATH = "data/fashion_queries.csv"
AVG_COSINE_SIMIL_RESULT = "evaluation/avg_cosine_similarity/results"

# === Models ===
IMAGE_TEXT_MODEL = "patrickjohncyh/fashion-clip"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# === Batch Sizes ===
TEXT_BATCH_SIZE = 512
IMAGE_BATCH_SIZE = 64

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DataLoader Settings ===
NUM_WORKERS = min(32, os.cpu_count())
PIN_MEMORY = DEVICE.type == "cuda"

# === Milvus Sizes ===
EMB_DIM = 512
INS_BATCH_SIZE = 10000
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# === LLM ===
LLM_MODEL = "llama3.1:8b"

# === API ===
API_URL = "http://127.0.0.1:8000/search/" 
TOP_K = 10 
