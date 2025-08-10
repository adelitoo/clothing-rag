import pandas as pd
from tqdm import tqdm
import logging
import json
import ollama
from evaluation.strategy import EvaluationStrategy
from ...src.fashion_search.core.config import settings

class LlmJudgeStrategy(EvaluationStrategy):
    def _load_prompt(self, prompt_path: str) -> str:
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"❌ Prompt file not found at: {prompt_path}")
            raise

    def _format_results(self, results: list, articles_df: pd.DataFrame) -> str:
        if not results: return "No results found.\n"
        ids = [int(item['article_id']) for item in results]
        details = articles_df[articles_df['article_id'].isin(ids)].set_index('article_id')
        formatted_str = ""
        for i, item in enumerate(results):
            prod_name = details.loc[int(item['article_id']), 'prod_name']
            formatted_str += f"{i+1}. {prod_name} (Score: {item['score']:.4f})\n"
        return formatted_str

    def _get_judgment(self, prompt: str, query: str) -> dict:
        try:
            response = ollama.chat(
                model=settings.LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                format="json"
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            logging.error(f"LLM Judge failed for query '{query}': {e}")
            return {"preference": "Error", "reasoning": str(e)}

    def execute(self):
        logging.info("🚀 EXECUTING STRATEGY: LLM-as-a-Judge A/B Evaluation...")

        prompt_template = self._load_prompt(settings.PROMPTS_DIR / "llm_judge_prompt.txt")
        articles_df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH)
        with open(settings.QUERIES_FILE_PATH, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]

        SYSTEM_A_ID = "vector_search"
        SYSTEM_B_ID = "baseline"
        system_a_name = settings.SYSTEMS_TO_EVALUATE[SYSTEM_A_ID]['name']
        system_b_name = settings.SYSTEMS_TO_EVALUATE[SYSTEM_B_ID]['name']

        summary = []
        for query in tqdm(queries, desc="A/B Evaluating Queries"):
            results_a = self.client.get_search_results(SYSTEM_A_ID, query, settings.EVALUATION_K)
            results_b = self.client.get_search_results(SYSTEM_B_ID, query, settings.EVALUATION_K)

            formatted_a = self._format_results(results_a, articles_df)
            formatted_b = self._format_results(results_b, articles_df)

            prompt = prompt_template.format(
                query=query, results_a=formatted_a, results_b=formatted_b,
                system_a_name=system_a_name, system_b_name=system_b_name
            )
            judgement = self._get_judgment(prompt, query)
            summary.append({"query": query, "llm_preference": judgement['preference'], "llm_reasoning": judgement['reasoning']})

        eval_df = pd.DataFrame(summary)
        preferences = eval_df['llm_preference'].value_counts(normalize=True).reindex(['A', 'B', 'Tie']).fillna(0) * 100
        print("\n" + preferences.to_string(float_format="%.1f%%"))

        self.reporting.save_dataframe_to_csv(eval_df, "llm_judge_detailed_report.csv", settings.REPORTS_DIR)
        self._plot_summary(eval_df)