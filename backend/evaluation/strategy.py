from abc import ABC, abstractmethod
import src.fashion_search.core.config as settings
from evaluation.core.api_client import SearchClient
from evaluation.core import metrics, reporting


class EvaluationStrategy(ABC):
    def __init__(self, client: SearchClient):
        self.client = client
        self.config = settings
        self.metrics = metrics
        self.reporting = reporting

    @abstractmethod
    def execute(self):
        pass
