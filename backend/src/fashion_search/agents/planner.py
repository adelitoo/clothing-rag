import json
from llama_index.llms.ollama import Ollama
from ..schemas.agent_schemas import OutfitPlan
from ..core.config import settings 

class OutfitPlanner:
    def __init__(self, llm: Ollama):
        self.llm = llm
        try:
            prompt_path = settings.PROMPTS_DIR / "outfit_planner_prompt.txt"
            self.planning_prompt_template = prompt_path.read_text()
            print("✅ Outfit planner prompt loaded successfully.")
        except FileNotFoundError:
            print(f"❌ FATAL: Outfit planner prompt not found at {prompt_path}")
            raise

    def analyze_query(self, query: str) -> OutfitPlan:
        planning_prompt = self.planning_prompt_template.format(query=query)
        try:
            response = self.llm.complete(planning_prompt)
            response_text = str(response).strip()

            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                return OutfitPlan(
                    categories=plan_data.get("categories", []),
                    descriptions=plan_data.get("descriptions", []),
                    is_single_item=plan_data.get("is_single_item", False),
                )
            else:
                return self._fallback_planning(query)
        except Exception as e:
            print(f"⚠️ Planning failed, using fallback: {e}")
            return self._fallback_planning(query)

    def _fallback_planning(self, query: str) -> OutfitPlan:
        query_lower = query.lower()
        single_item_keywords = ["jeans", "shirt", "dress", "shoes", "jacket", "pants", "top"]
        
        if any(keyword in query_lower for keyword in single_item_keywords) and "outfit" not in query_lower:
            category = "dress" 
            if "jeans" in query_lower or "pants" in query_lower: category = "pants"
            elif "shirt" in query_lower: category = "shirt"
            return OutfitPlan([category], [query], True)

        return OutfitPlan(
            ["shirt", "pants", "shoes"],
            [f"{query} shirt", f"{query} pants", f"{query} shoes"],
            False,
        )