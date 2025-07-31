import numpy as np
from typing import List, Tuple, Set

def calculate_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(relevance_scores / discounts)

def calculate_ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    dcg = calculate_dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg_at_k(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_average_precision_at_k(relevance_scores: List[float], k: int, threshold: float) -> float:
    relevance_scores = np.asarray(relevance_scores)[:k]
    binary_relevance = (relevance_scores >= threshold).astype(int)
    
    total_relevant_items = np.sum(binary_relevance)
    if total_relevant_items == 0:
        return 0.0
        
    precision_at_k = np.cumsum(binary_relevance) / (np.arange(len(binary_relevance)) + 1)
    return np.sum(precision_at_k * binary_relevance) / total_relevant_items

def calculate_reciprocal_rank_at_k(relevance_scores: List[float], k: int, threshold: float) -> float:
    relevance_scores = np.asarray(relevance_scores)[:k]
    binary_relevance = (relevance_scores >= threshold).astype(int)
    first_relevant_indices = np.where(binary_relevance == 1)[0]
    
    if len(first_relevant_indices) > 0:
        first_rank = first_relevant_indices[0] + 1
        return 1.0 / first_rank
    
    return 0.0

def calculate_precision_recall_f1_at_k(
    retrieved_ids: List[int], relevant_ids: Set[int], k: int
) -> Tuple[float, float, float]:
    if k == 0:
        return 0.0, 0.0, 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    
    true_positives = len(retrieved_at_k.intersection(relevant_ids))
    
    precision = true_positives / k
    
    recall = true_positives / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
    
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1_score