import pandas as pd
from tqdm import tqdm
import logging
import os
from evaluation.strategy import EvaluationStrategy
from ...src.fashion_search.core.config import settings

class AnnotationCreationStrategy(EvaluationStrategy):
    def _get_image_path(self, article_id: int) -> str:
        padded_id = str(article_id).zfill(10)
        return os.path.abspath(settings.IMAGE_BASE_DIR / padded_id[:3] / f"{padded_id}.jpg")

    def _create_hyperlink(self, path: str) -> str:
        return f'=HYPERLINK("file:///{path}", "View Image")'

    def execute(self):
        logging.info("🚀 EXECUTING STRATEGY: Annotation File Creation...")
        with open(settings.QUERIES_FILE_PATH, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        articles_df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH, dtype={'article_id': int})

        records = []
        seen_pairs = set()
        for query in tqdm(queries, desc="Pooling results"):
            pooled_ids = set()
            for system_id in settings.SYSTEMS_TO_EVALUATE.keys():
                results = self.client.get_search_results(system_id, query, settings.EVALUATION_K)
                pooled_ids.update(item['article_id'] for item in results)

            for article_id in pooled_ids:
                if (query, article_id) not in seen_pairs:
                    records.append({"query": query, "article_id": article_id})
                    seen_pairs.add((query, article_id))

        annotation_df = pd.DataFrame(records)
        enriched_df = pd.merge(annotation_df, articles_df[['article_id', 'prod_name', 'detail_desc']], on='article_id', how='left')
        enriched_df['image_link'] = enriched_df['article_id'].apply(self._get_image_path).apply(self._create_hyperlink)
        enriched_df['relevance'] = ''

        final_df = enriched_df[['query', 'article_id', 'prod_name', 'detail_desc', 'image_link', 'relevance']]
        output_path = settings.ANNOTATION_FILE_OUTPUT
        final_df.to_csv(output_path, index=False)
        logging.info(f"✅ Success! Annotation file created at: '{output_path}'")