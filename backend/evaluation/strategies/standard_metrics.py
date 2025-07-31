import pandas as pd
from tqdm import tqdm
import logging
from evaluation.strategy import EvaluationStrategy

class StandardMetricsStrategy(EvaluationStrategy):
    def execute(self):
        logging.info("🚀 EXECUTING STRATEGY: Standard Metrics (Precision, Recall, F1)...")

        gt_df = pd.read_csv(self.config.GROUND_TRUTH_FILE)
        relevant_df = gt_df[gt_df['relevance'] >= self.config.RELEVANCE_THRESHOLD]
        relevant_items_per_query = relevant_df.groupby('query')['article_id'].apply(set).to_dict()
        unique_queries = gt_df['query'].unique()

        results_data = []
        for query in tqdm(unique_queries, desc="Evaluating Queries"):
            relevant_ids = relevant_items_per_query.get(query, set())
            for system_id, system_info in self.config.SYSTEMS_TO_EVALUATE.items():
                ranked_results = self.client.get_search_results(system_id, query, self.config.EVALUATION_K)
                ranked_ids = [item['article_id'] for item in ranked_results]

                precision, recall, f1 = self.metrics.calculate_precision_recall_f1_at_k(
                    ranked_ids, relevant_ids, self.config.EVALUATION_K
                )
                results_data.append({
                    "system_name": system_info['name'],
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                })

        results_df = pd.DataFrame(results_data)
        summary_df = results_df.groupby('system_name')[['Precision', 'Recall', 'F1-Score']].mean().reset_index()
        summary_for_plot = summary_df.rename(columns={'system_name': 'Metric'}).set_index('Metric').T.reset_index().rename(columns={'index': 'Metric'})

        system_names_for_print = [sys_info['name'] for sys_info in self.config.SYSTEMS_TO_EVALUATE.values()]
        summary_for_print = summary_df.melt(id_vars='system_name', var_name='Metric', value_name='Score').pivot(index='Metric', columns='system_name', values='Score')[system_names_for_print]

        print("\n" + summary_for_print.to_string(float_format="%.4f"))
        self.reporting.plot_grouped_bar_chart(
            summary_df=summary_for_plot,
            systems_config=self.config.SYSTEMS_TO_EVALUATE,
            title='Standard Metrics Performance Comparison',
            output_filename='standard_metrics_barchart.png',
            reports_dir=self.config.REPORTS_DIR
        )