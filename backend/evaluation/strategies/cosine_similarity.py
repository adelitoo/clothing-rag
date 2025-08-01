import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from evaluation.strategy import EvaluationStrategy

class CosineSimilarityStrategy(EvaluationStrategy):
    def _get_average_similarity(self, results: list) -> float:
        if not results: return 0.0
        return np.mean([item['score'] for item in results])

    def execute(self):
        logging.info("🚀 EXECUTING STRATEGY: Average Cosine Similarity...")
        with open(self.config.QUERIES_FILE_PATH, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]

        results_data = []
        for query in tqdm(queries, desc="Evaluating Queries"):
            for system_id, system_info in self.config.SYSTEMS_TO_EVALUATE.items():
                results = self.client.get_search_results(system_id, query, self.config.EVALUATION_K)
                avg_score = self._get_average_similarity(results)
                results_data.append({'model': system_info['name'], 'score': avg_score})

        results_df = pd.DataFrame(results_data)
        for name in results_df['model'].unique():
            overall_avg = results_df[results_df['model'] == name]['score'].mean()
            logging.info(f"📈 {name:<20} | Overall Avg. Similarity: {overall_avg:.4f}")

        self.reporting.plot_score_distribution(
            results_df, self.config.SYSTEMS_TO_EVALUATE,
            f'Cosine Similarity Distributions (Top {self.config.EVALUATION_K})',
            'cosine_similarity_distribution.png', self.config.REPORTS_DIR
        )