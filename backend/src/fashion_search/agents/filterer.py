import json
from llama_index.llms.ollama import Ollama
from ..core.config import settings

class FilterAgent:
    def __init__(self, llm: Ollama):
        self.llm = llm
        try:
            prompt_path = settings.PROMPTS_DIR / "filter_extractor_prompt.txt"
            self.prompt_template = prompt_path.read_text()
        except FileNotFoundError:
            print(f"❌ FATAL: Filter extractor prompt not found at {prompt_path}")
            raise

    def extract_filters(self, query: str) -> dict:
        prompt = self.prompt_template.format(query=query)
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            print(f"⚠️ Filter extraction failed: {e}")
            return {} 