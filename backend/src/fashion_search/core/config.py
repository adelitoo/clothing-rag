import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.EVALUATION_DIR = self.PROJECT_ROOT / "evaluation"
        self.IMAGE_BASE_DIR = self.DATA_DIR / "images"  
        self.REPORTS_DIR = self.EVALUATION_DIR / "reports"
        self.PROMPTS_DIR = self.PROJECT_ROOT / "prompts"

        self.ARTICLES_CSV_PATH = self.DATA_DIR / "articles.csv"
        self.COMPLETE_ARTICLES_CSV_PATH = self.DATA_DIR / "complete_articles.csv"
        self.EMBEDDING_SAVE_PATH = self.DATA_DIR / "embeddings.npz"
        self.QUERIES_FILE_PATH = self.DATA_DIR / "fashion_queries.csv"
        self.GROUND_TRUTH_FILE = self.DATA_DIR / "ground_truth.csv"
        self.ANNOTATION_FILE_OUTPUT = self.REPORTS_DIR / "to_annotate.csv"


        self.IMAGE_TEXT_MODEL = "patrickjohncyh/fashion-clip"
        self.IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
        self.LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "qwen2.5:7b-instruct-q8_0")

        self.TEXT_BATCH_SIZE = 512
        self.IMAGE_BATCH_SIZE = 64
        self._setup_device_settings()

        self.EMB_DIM = 512
        self.MILVUS_INSERT_BATCH_SIZE = 10000
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = os.getenv("REDIS_PORT", "6379")

        self.EVALUATION_K = 10
        self.RELEVANCE_THRESHOLD = 2
        self.SYSTEMS_TO_EVALUATE = {
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

    def _setup_device_settings(self):
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")


settings = Settings()