"""Define the common interface and ranking utilities for retrieval systems."""

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple, overload

import numpy as np


class Retriever(ABC):
    """Abstract base class for indexing and retrieving ranked documents."""

    @staticmethod
    def is_descending(scores: List[float]) -> bool:
        """Return whether scores are ordered from highest to lowest."""
        return all(earlier >= later for earlier, later in zip(scores, scores[1:]))

    @staticmethod
    def get_rank(id_list: List[int], score_list: List[float]) -> Dict[int, int]:
        """Map document IDs to one-based ranks derived from descending scores."""
        return {x: i for i, (s, x) in enumerate(sorted(zip(score_list, id_list), reverse=True), start=1)}

    @staticmethod
    def distance2score(distances: List[float]) -> List[float]:
        """Convert distances into monotonically decreasing similarity scores."""
        distances_vec = np.array(distances)
        alpha = np.mean(distances_vec)
        scores = np.exp(-alpha * np.square(distances_vec))
        return scores.tolist()

    @staticmethod
    def rrf(ranks: List[List[Dict[int, int]]], k=1000) -> Tuple[List[List[int]], List[List[float]]]:
        """Fuse multiple rankers with reciprocal rank fusion."""
        final_results = []
        final_scores = []
        for query_idx in range(len(ranks[0])):
            all_id_set = set()
            for ranker in ranks:
                all_id_set |= set(ranker[query_idx].keys())
            results_with_scores = []
            for pid in all_id_set:
                rrf_score = 0
                for ranker in ranks:
                    if pid in ranker[query_idx]:
                        rrf_score += 1 / (60 + ranker[query_idx][pid])
                results_with_scores.append((pid, rrf_score))
            sorted_results = [
                (pid, score) for pid, score in sorted(results_with_scores, key=lambda x: x[1], reverse=True)
            ]
            final_result = [pid for pid, score in sorted_results]
            final_score = [score for pid, score in sorted_results]
            final_results.append(final_result[:k])
            final_scores.append(final_score[:k])
        return final_results, final_scores

    @abstractmethod
    def index(self, corpus: List[str], pids: List[int]) -> None:
        """Build an index for the provided corpus and document IDs."""
        raise NotImplementedError("Subclasses must implement the index method")

    @overload
    def retrieve(
        self,
        qids: List[int],
        queries: List[str],
        pids: List[int],
        paragraphs: List[str],
        k=100,
        click: Literal[False] = False,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Retrieve ranked documents without click scores."""
        ...

    @overload
    def retrieve(
        self,
        qids: List[int],
        queries: List[str],
        pids: List[int],
        paragraphs: List[str],
        k=100,
        click: Literal[True] = True,
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """Retrieve ranked documents together with click scores."""
        ...

    @abstractmethod
    def retrieve(
        self,
        qids: List[int],
        queries: List[str],
        pids: List[int],
        paragraphs: List[str],
        k=100,
        click: bool = False,
    ) -> Tuple[List[List[int]], List[List[float]]] | Tuple[List[List[int]], List[List[float]], List[float]]:
        """Retrieve ranked documents and scores for each query."""
        raise NotImplementedError("Subclasses must implement the retrieve method")
