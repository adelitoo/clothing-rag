import pandas as pd
from tqdm import tqdm
import logging
from evaluation.strategy import EvaluationStrategy
from src.fashion_search.core.config import settings


class RankingMetricsStrategy(EvaluationStrategy):
    def execute(self):
        logging.info("🚀 EXECUTING STRATEGY: Ranking Metrics (nDCG, MAP, MRR)...")

        gt_df = pd.read_csv(settings.GROUND_TRUTH_FILE)
        gt_df.dropna(subset=["relevance"], inplace=True)
        gt_df["relevance"] = gt_df["relevance"].astype(int)
        relevance_map = {
            (row.query, row.article_id): row.relevance for row in gt_df.itertuples()
        }
        unique_queries = gt_df["query"].unique()

        results_data = []
        for query in tqdm(unique_queries, desc="Evaluating Queries"):
            for system_id, system_info in settings.SYSTEMS_TO_EVALUATE.items():
                ranked_results = self.client.get_search_results(
                    system_id, query, settings.EVALUATION_K # Use settings directly
                )
                ranked_ids = [item["article_id"] for item in ranked_results]
                relevance_scores = [
                    relevance_map.get((query, aid), 0) for aid in ranked_ids
                ]

                ndcg = self.metrics.calculate_ndcg_at_k(
                    relevance_scores, settings.EVALUATION_K
                )
                map_score = self.metrics.calculate_average_precision_at_k(
                    relevance_scores,
                    settings.EVALUATION_K,
                    settings.RELEVANCE_THRESHOLD,
                )
                mrr = self.metrics.calculate_reciprocal_rank_at_k(
                    relevance_scores,
                    settings.EVALUATION_K,
                    settings.RELEVANCE_THRESHOLD,
                )

                results_data.append(
                    {
                        "system_name": system_info["name"],
                        "nDCG": ndcg,
                        "MAP": map_score,
                        "MRR": mrr,
                    }
                )

        results_df = pd.DataFrame(results_data)
        summary_df = (
            results_df.groupby("system_name")[["nDCG", "MAP", "MRR"]]
            .mean()
            .reset_index()
        )
        summary_for_plot = (
            summary_df.melt(
                id_vars="system_name", var_name="Metric", value_name="Score"
            )
            .pivot(index="Metric", columns="system_name", values="Score")
            .reset_index()
        )

        print("\n" + summary_df.to_string(index=False, float_format="%.4f"))
        self.reporting.plot_performance_heatmap(
            summary_df=summary_for_plot,
            title="Ranking Metrics Performance Comparison",
            output_filename="ranking_metrics_heatmap.png",
            reports_dir=settings.REPORTS_DIR,
        )