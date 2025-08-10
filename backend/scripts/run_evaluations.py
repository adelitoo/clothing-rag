import argparse
from src.fashion_search.core.config import settings
from evaluation.core.api_client import SearchClient
from evaluation.strategies.ranking_metrics import RankingMetricsStrategy
from evaluation.strategies.standard_metrics import StandardMetricsStrategy
from evaluation.strategies.llm_judge import LlmJudgeStrategy
from evaluation.strategies.cosine_similarity import CosineSimilarityStrategy
from evaluation.strategies.annotation_creation import AnnotationCreationStrategy


def main():
    strategy_map = {
        "ranking": RankingMetricsStrategy,
        "standard": StandardMetricsStrategy,
        "judge": LlmJudgeStrategy,
        "similarity": CosineSimilarityStrategy,
        "annotate": AnnotationCreationStrategy,
    }

    parser = argparse.ArgumentParser(
        description="Run evaluation strategies for the fashion search system."
    )
    parser.add_argument(
        "strategy", choices=strategy_map.keys(), help="The evaluation strategy to run."
    )
    args = parser.parse_args()

    client = SearchClient(settings.SYSTEMS_TO_EVALUATE)
    SelectedStrategyClass = strategy_map[args.strategy]
    strategy_instance = SelectedStrategyClass(client)
    strategy_instance.execute()


if __name__ == "__main__":
    main()
