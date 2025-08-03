from dataclasses import dataclass
from typing import List

@dataclass
class SearchResult:
    """Represents a single search result item."""
    article_id: str
    score: float
    category: str

@dataclass
class OutfitPlan:
    """Represents the multi-step search plan created by the agent."""
    categories: List[str]
    descriptions: List[str]
    is_single_item: bool