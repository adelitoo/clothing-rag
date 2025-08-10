from llama_index.llms.ollama import Ollama
from ..core.config import settings
from typing import List, Dict

class QueryRefinementAgent:
    def __init__(self, llm: Ollama):
        self.llm = llm
        try:
            prompt_path = settings.PROMPTS_DIR / "query_refinement_prompt.txt"
            self.prompt_template = prompt_path.read_text()
            print("✅ QueryRefinementAgent initialized.")
        except FileNotFoundError:
            print(f"❌ FATAL: Query refinement prompt not found at {prompt_path}")
            raise

    def refine(self, new_query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return new_query

        history_str = "\n".join([f"- {turn['role']}: {turn['content']}" for turn in history])
        
        prompt = self.prompt_template.format(history=history_str, new_query=new_query)
        
        print(f"🧠 Refining query with history. Latest query: '{new_query}'")

        try:
            response = self.llm.complete(prompt)
            refined_query = str(response).strip()
            print(f"✨ Refined query: '{refined_query}'")
            return refined_query
        except Exception as e:
            print(f"⚠️ Query refinement failed, using original query. Error: {e}")
            return new_query