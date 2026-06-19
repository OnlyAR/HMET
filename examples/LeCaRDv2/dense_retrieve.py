"""Retrieve and rank LeCaRDv2 candidates with FAISS dense embeddings."""

import json
import re
from pathlib import Path
from typing import List, Tuple, TypedDict

import click
import faiss
from loguru import logger
import numpy as np
import torch

from hierarchical_pipeline import metric_single


dim_dict = {
    "qwen3-embedding-0.6b": 1024,
}


class ResultItem(TypedDict):
    pid: int
    score: float


def get_model_name_from_path(embedding_path: str) -> str:
    """Extract the embedding model name from an embedding file path."""
    pattern = re.compile(r"(qwen3-embedding-.+b)")
    match = pattern.search(embedding_path)
    if match:
        return match.group(1)
    raise ValueError("Could not extract the model name from the embedding path.")


def load_embeddings(embedding_path: str) -> Tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Load query and candidate embeddings from a PyTorch checkpoint."""
    embedding_dict = torch.load(embedding_path)
    query_embeddings = embedding_dict["query_embeddings"]
    candidate_embeddings = embedding_dict["candidate_embeddings"]
    return query_embeddings, candidate_embeddings


def embeddings_to_list(embeddings: dict[int, torch.Tensor]) -> Tuple[List[int], np.ndarray]:
    """Convert an ID-to-tensor mapping into aligned IDs and a NumPy matrix."""
    ids = []
    emb_list = []
    for pid, emb in embeddings.items():
        ids.append(pid)
        emb_list.append(emb.numpy())
    return ids, np.array(emb_list)


def distance_to_score(distances: List[float]) -> List[float]:
    """Apply the Gaussian kernel from the paper to squared L2 distances.

    ``faiss.IndexFlatL2`` returns squared Euclidean distances, so the
    paper's ``exp(-||q-d||_2^2)`` score is obtained directly with
    ``exp(-distance)``.
    """
    distances_vec = np.array(distances)
    scores = np.exp(-distances_vec)
    return scores.tolist()


def faiss_retrieve(
    query_embeddings: dict[int, torch.Tensor],
    candidate_embeddings: dict[int, torch.Tensor],
    model_name: str,
) -> dict[int, List[ResultItem]]:
    """Retrieve all candidates for each query using a FAISS L2 index."""
    dim = dim_dict[model_name]
    database = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    pids, p_embeddings = embeddings_to_list(candidate_embeddings)

    database.add_with_ids(np.array(p_embeddings), np.array(pids))  # pyright: ignore[reportCallIssue]

    qids, q_embeddings = embeddings_to_list(query_embeddings)

    distances, indices = database.search(q_embeddings, len(pids))  # pyright: ignore[reportCallIssue]
    results: dict[int, List[ResultItem]] = {}
    for qid, indice, distance in zip(qids, indices, distances):
        result_items: List[ResultItem] = []
        scores = distance_to_score(distance)
        for pid, score in zip(indice, scores):
            item: ResultItem = {"pid": int(pid), "score": float(score)}
            result_items.append(item)
        results[qid] = result_items
    return results


def dense_retrieve(embedding_path: str) -> dict[int, List[ResultItem]]:
    """Load embeddings and produce ranked candidate lists for every query."""
    model_name = get_model_name_from_path(embedding_path)
    logger.info(f"Detected model name: {model_name}")
    query_embeddings, candidate_embeddings = load_embeddings(embedding_path)
    logger.info(f"Loaded {len(query_embeddings)} query embeddings and {len(candidate_embeddings)} candidate embeddings.")
    results = faiss_retrieve(query_embeddings, candidate_embeddings, model_name)
    logger.info("Retrieval completed.")
    return results


def debug_retrieve_results(results: dict[int, List[ResultItem]], top_k: int = 5) -> dict[int, List[ResultItem]]:
    """Keep only the top-ranked results for compact debugging output."""
    debug_results = {}
    for qid, items in results.items():
        debug_results[qid] = items[:top_k]
    return debug_results


def save_results(results, embedding_path, debug: bool = False) -> Path:
    """Save retrieval results and, optionally, a compact debug result file."""
    embedding_path = Path(embedding_path)
    result_path = Path("gear_results").joinpath(*embedding_path.parts[1:]).with_suffix(".json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved retrieval results to: {result_path}")

    if debug:
        debug_results = debug_retrieve_results(results)
        debug_result_path = result_path.with_name(result_path.stem + "_debug.json")
        debug_result_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_result_path.open("w", encoding="utf-8") as f:
            json.dump(debug_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved debug retrieval results to: {debug_result_path}")
    return result_path


@click.command()
@click.option("--path", type=str, required=True, help="Path to the embeddings.")
def main(path):
    """Run dense retrieval from the command line and evaluate the results."""
    results = dense_retrieve(path)
    result_path = save_results(results, path, debug=False)
    metric_single(path=result_path)


if __name__ == "__main__":
    main()
