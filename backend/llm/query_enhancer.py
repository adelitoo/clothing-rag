import ollama
from functools import lru_cache
from pathlib import Path

class LLMQueryEnhancer:
    def __init__(self, model: str, prompt_dir: Path):
        self.model = model
        self.transform_prompt = self._load_prompt(prompt_dir / "transform_query_system.txt")
        self.summarize_template = self._load_prompt(prompt_dir / "summarize_query.txt")

    def _load_prompt(self, file_path: Path) -> str:
        try:
            return file_path.read_text()
        except FileNotFoundError:
            print(f"❌ Prompt file not found: {file_path}")
            raise

    def _execute_chat(self, messages: list) -> str:
        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response['message']['content'].strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "" 

    @lru_cache(maxsize=256)
    def transform(self, user_query: str) -> str:
        if not user_query:
            return ""
        
        messages = [
            {"role": "system", "content": self.transform_prompt},
            {"role": "user", "content": user_query}
        ]
        return self._execute_chat(messages)

    @lru_cache(maxsize=256)
    def summarize(self, transformed_query: str) -> str:
        if not transformed_query:
            return ""
            
        prompt = self.summarize_template.format(transformed_query=transformed_query)
        messages = [{"role": "user", "content": prompt}]
        return self._execute_chat(messages)
