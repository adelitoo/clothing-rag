from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class SearchResult:
    article_id: str
    score: float
    category: str

@dataclass
class OutfitPlan:
    categories: List[str]
    descriptions: List[str]
    is_single_item: bool
    filters: Dict[str, str] = field(default_factory=dict)

class RecommendedArticle(BaseModel):
    article_id: str
    relevance_score: float

class FormattedResponse(BaseModel):
    summary_text: str = Field(description="A friendly, user-facing summary formatted with markdown lists.")
    recommended_articles: List[RecommendedArticle]