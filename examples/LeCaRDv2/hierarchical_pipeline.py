"""Fuse and evaluate multi-view LeCaRDv2 retrieval results."""

import json
from pathlib import Path
from typing import Dict, List, TypedDict

import click
from loguru import logger

from loaders import load_label_list
from utils.metric import MRR_mls, recall_at_1, recall_at_5, recall_at_10, recall_at_20


class ResultItem(TypedDict):
    pid: int
    score: float


def filter_test_queries(
    results: Dict[int, List[ResultItem]],
    test_q_keys: List[int],
) -> Dict[int, List[ResultItem]]:
    """Keep retrieval results whose query IDs belong to the test set."""
    return {k: v for k, v in results.items() if k in test_q_keys}


def hierarchical_retrieve(
    test_results: List[Dict[int, List[ResultItem]]],
    topks: List[int],
    weights: List[float],
):
    """Apply sequential weighted retrieval with top-k filtering at each stage."""
    final_results: Dict[int, List[ResultItem]] = {}
    for test_result, topk, weight in zip(test_results, topks, weights):
        for qid, items in test_result.items():
            last_pid_score_dict = {item["pid"]: item["score"] for item in final_results.get(qid, [])}
            current_pid_score_dict = {}
            for item in items:
                pid = item["pid"]
                if last_pid_score_dict and pid not in last_pid_score_dict:
                    continue
                current_pid_score_dict[pid] = last_pid_score_dict.get(pid, 0) + item["score"] * weight
            current_list = [ResultItem(pid=pid, score=score) for pid, score in current_pid_score_dict.items()]
            current_list.sort(key=lambda x: x["score"], reverse=True)
            final_results[qid] = current_list[:topk]
    return final_results


def weighted_score_retrieve(
    test_results: List[Dict[int, List[ResultItem]]],
    weights: List[float],
    topk: int,
) -> Dict[int, List[ResultItem]]:
    """Fuse Gaussian view similarities using the paper's weighted sum."""
    score_dct: Dict[int, Dict[int, float]] = {}

    for test_result, weight in zip(test_results, weights):
        for qid, items in test_result.items():
            if qid not in score_dct:
                score_dct[qid] = {}

            for item in items:
                pid = item["pid"]
                if pid not in score_dct[qid]:
                    score_dct[qid][pid] = 0.0
                score_dct[qid][pid] += item["score"] * weight

    # Sort by the fused score and keep the top-k candidates.
    final_results: Dict[int, List[ResultItem]] = {}
    for qid, pid_score_dct in score_dct.items():
        sorted_items = sorted(pid_score_dct.items(), key=lambda x: x[1], reverse=True)
        final_results[qid] = [ResultItem(pid=pid, score=score) for pid, score in sorted_items[:topk]]
    return final_results


def load_test_results(paths: List[str], pids: List[int]) -> List[Dict[int, List[ResultItem]]]:
    """Load retrieval result files and retain only the requested query IDs."""
    all_results: List[Dict[int, List[ResultItem]]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
            int_result = {int(k): v for k, v in result.items()}
            all_results.append(filter_test_queries(int_result, pids))
    return all_results


def print_metrics(test_qids: List[int], results: Dict[int, List[ResultItem]], labels: List[List[int]]):
    """Compute and log recall and MRR metrics for ranked retrieval results."""
    preds = []
    for qid in test_qids:
        pred_items = results[qid]
        pred_pids = [int(item["pid"]) for item in pred_items]
        preds.append(pred_pids[:30])

    r1 = recall_at_1(labels, preds)
    r5 = recall_at_5(labels, preds)
    r10 = recall_at_10(labels, preds)
    r20 = recall_at_20(labels, preds)
    mrr = MRR_mls(labels, preds)
    logger.info(f"Recall@1: {r1:.4f}")
    logger.info(f"Recall@5: {r5:.4f}")
    logger.info(f"Recall@10: {r10:.4f}")
    logger.info(f"Recall@20: {r20:.4f}")
    logger.info(f"MRR@20: {mrr:.4f}")
    return {"Recall@1": r1, "Recall@5": r5, "Recall@10": r10, "Recall@20": r20, "MRR": mrr}


root_path = Path(__file__).parent.parent.parent
data_path = root_path / "data" / "LeCaRDv2"
test_list = sorted(set(map(int, json.load(open(data_path / "test_qids.json", "r", encoding="utf-8")))))


def metric_single(path):
    """Evaluate a single retrieval result file on the test queries."""
    test_qids = test_list
    result = load_test_results(
        paths=[path],
        pids=test_qids,
    )[0]
    labels = load_label_list(test_qids)
    return print_metrics(test_qids, result, labels)


@click.command()
@click.option("--model", default="qwen3-embedding-0.6b", show_default=True, help="Embedding model name.")
@click.option("--subject-weight", type=click.FloatRange(min=0), required=True, help="Weight for the subject view.")
@click.option(
    "--subjective-element-weight",
    type=click.FloatRange(min=0),
    required=True,
    help="Weight for the subjective-element view.",
)
@click.option("--object-weight", type=click.FloatRange(min=0), required=True, help="Weight for the object view.")
@click.option(
    "--objective-element-weight",
    type=click.FloatRange(min=0),
    required=True,
    help="Weight for the objective-element view.",
)
def main(
    model: str,
    subject_weight: float,
    subjective_element_weight: float,
    object_weight: float,
    objective_element_weight: float,
):
    """Fuse four manually weighted fact views and evaluate the final ranking."""
    facts = ["subject", "subjective_element", "object", "objective_element"]
    weights = [
        subject_weight,
        subjective_element_weight,
        object_weight,
        objective_element_weight,
    ]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise click.UsageError("The four view weights must sum to 1.")

    original_result_path = Path(f"gear_results/origin/embeddings_{model}.json")
    fact_result_paths = [Path(f"gear_results/fact/{fact}/embeddings_{model}.json") for fact in facts]

    original_result = load_test_results(paths=[str(original_result_path)], pids=test_list)[0]
    l1_results = []
    for fact_result_path in fact_result_paths:
        fact_result = load_test_results(paths=[str(fact_result_path)], pids=test_list)[0]
        l1_results.append(
            hierarchical_retrieve(
                test_results=[original_result, fact_result],
                topks=[200, 200],
                weights=[0, 1],
            )
        )

    labels = load_label_list(qids=test_list)

    log_path = root_path / "logs" / "hierarchical_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="10 MB")

    final_results = weighted_score_retrieve(
        test_results=l1_results,
        weights=weights,
        topk=30,
    )
    logger.info(f"Evaluating manually specified weights: {weights}")

    metric = print_metrics(test_list, final_results, labels)
    formatted_metric = {key: f"{value:.4f}" for key, value in metric.items()}
    logger.info(f"Weights: {weights}, metric sum: {sum(metric.values()):.4f}, metrics: {formatted_metric}")
    with open("best_metric.txt", "a", encoding="utf-8") as f:
        f.write("\t".join(formatted_metric.values()) + "\n")


if __name__ == "__main__":
    main()
