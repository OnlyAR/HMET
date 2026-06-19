import json
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

root_path = Path(__file__).parent.parent.parent
data_path: Path = root_path / "data" / "LeCaRDv2"

candidate_path = data_path / "candidates.jsonl"
query_path = data_path / "test_queries.jsonl"
label_path = data_path / "labels.json"


def load_test_cand_data() -> Tuple[List, List]:
    queries = []
    with query_path.open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    candidates = []
    with candidate_path.open(encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    return queries, candidates


def load_cases(path) -> List[Dict]:
    """
    [{"id": 1, "fact": "瑞安市人民检察院指控..."}, ...]
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cases.append(item)
    logger.debug(f"Loaded {len(cases)} cases from {path}")
    return cases


def load_labels() -> Dict[int, List[int]]:
    """
    {1: [111, 222, 333], ...}
    """
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        origin_labels = json.load(f)
    for k, pairs in origin_labels.items():
        labels[int(k)] = [int(pair[0]) for pair in pairs]
    return labels


def load_label_list(qids: List[int]) -> List[List[int]]:
    label_dict = load_labels()
    labels = []
    for qid in qids:
        golds = []
        pairs = label_dict[qid]
        for pid in pairs:
            golds.append(pid)
        labels.append(golds)
    return labels
