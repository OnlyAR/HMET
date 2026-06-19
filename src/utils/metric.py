"""Provide ranking and classification metrics used by retrieval experiments."""

import numpy as np
import torch


def accuracy(output, target):
    """Compute classification accuracy from model logits and target labels."""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def _recall(target_lists: list, pred_lists: list, at_k: int):
    """Compute mean recall at a fixed cutoff.

    Args:
        target_lists: Relevant document IDs for each query.
        pred_lists: Ranked predicted document IDs for each query.
        at_k: Number of top-ranked predictions to consider.
    """

    def _recall_per_user(t_list, p_list):
        """Compute recall for one query."""
        return len(set(t_list) & set(p_list)) / len(t_list) if len(t_list) > 0 else 0.0

    assert len(target_lists) == len(pred_lists)
    recall_value = 0.0
    for target_list, pred_list in zip(target_lists, pred_lists):
        recall_per_user = _recall_per_user(target_list, pred_list[:at_k])
        recall_value += recall_per_user
    return recall_value / len(target_lists)


def _MRR(target_lists: list, pred_lists: list, mode="multi_labels"):
    """Compute mean reciprocal rank in multi-label or single-label mode.

    Args:
        target_lists: Relevant document IDs for each query.
        pred_lists: Ranked predicted document IDs for each query.
        mode: Either ``multi_labels`` or ``single_label``.
    """
    assert len(target_lists) == len(pred_lists)

    if mode == "multi_labels":
        mrr = 0.0
        for t_list, p_list in zip(target_lists, pred_lists):
            t_set = set(t_list)
            p_true = [1 if p in t_set else 0 for p in p_list]
            if sum(p_true) == 0:
                continue
            rr_score = p_true / (np.arange(len(p_true)) + 1)
            mrr += np.sum(rr_score) / np.sum(p_true)
        return mrr / len(target_lists)
    elif mode == "single_label":
        mrr = 0.0
        for t_list, p_list in zip(target_lists, pred_lists):
            target = t_list[0]
            if target not in p_list:
                continue
            mrr += 1 / (p_list.index(target) + 1)
        return mrr / len(target_lists)
    else:
        raise ValueError("ERROR: MRR mode must be multi_labels or single_label")


def recall_at_1(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 1."""
    return _recall(target_lists, pred_lists, 1)


def recall_at_5(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 5."""
    return _recall(target_lists, pred_lists, 5)


def recall_at_10(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 10."""
    return _recall(target_lists, pred_lists, 10)


def recall_at_20(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 20."""
    return _recall(target_lists, pred_lists, 20)


def recall_at_30(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 30."""
    return _recall(target_lists, pred_lists, 30)


def recall_at_100(target_lists: list, pred_lists: list):
    """Compute mean recall at rank 100."""
    return _recall(target_lists, pred_lists, 100)


def MRR_mls(target_lists: list, pred_lists: list):
    """Compute multi-label mean reciprocal rank."""
    return _MRR(target_lists, pred_lists, "multi_labels")


def MRR_sl(target_lists: list, pred_lists: list):
    """Compute single-label mean reciprocal rank."""
    return _MRR(target_lists, pred_lists, "single_label")


def precision_at_k(target_lists: list, pred_lists: list, k: int):
    """Compute mean precision at rank ``k``."""
    precisions = []
    for t_list, p_list in zip(target_lists, pred_lists):
        p_list_k = p_list[:k]
        if not p_list_k:
            precisions.append(0.0)
            continue
        correct = sum(1 for p in p_list_k if p in t_list)
        precisions.append(correct / k)
    return float(np.mean(precisions)) if precisions else 0.0


def ndcg_at_k(target_lists: list, pred_lists: list, k: int):
    """Compute mean normalized discounted cumulative gain at rank ``k``."""
    ndcgs = []
    for t_list, p_list in zip(target_lists, pred_lists):
        p_list_k = p_list[:k]
        dcg = 0.0
        for i, p in enumerate(p_list_k):
            if p in t_list:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 0.0
        for i in range(min(len(t_list), k)):
            idcg += 1.0 / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def mrr_at_k(target_lists: list, pred_lists: list, k: int):
    """Compute mean reciprocal rank at rank ``k``."""
    mrrs = []
    for t_list, p_list in zip(target_lists, pred_lists):
        p_list_k = p_list[:k]
        for i, p in enumerate(p_list_k):
            if p in t_list:
                mrrs.append(1.0 / (i + 1))
                break
        else:
            mrrs.append(0.0)
    return float(np.mean(mrrs)) if mrrs else 0.0


def ap(target_lists: list, pred_lists: list):
    """Compute mean average precision across queries."""
    aps = []
    for t_list, p_list in zip(target_lists, pred_lists):
        if not t_list:
            aps.append(0.0)
            continue
        correct = 0
        sum_prec = 0.0
        for i, p in enumerate(p_list):
            if p in t_list:
                correct += 1
                sum_prec += correct / (i + 1)
        aps.append(sum_prec / len(t_list) if len(t_list) > 0 else 0.0)
    return float(np.mean(aps)) if aps else 0.0


def micro_f1(target_lists: list, pred_lists: list, k: int):
    """Compute micro-averaged F1 at rank ``k`` across all queries."""
    tp = 0
    fp = 0
    fn = 0
    for t_list, p_list in zip(target_lists, pred_lists):
        p_list_k = set(p_list[:k])
        t_list_set = set(t_list)
        tp += len(p_list_k & t_list_set)
        fp += len(p_list_k - t_list_set)
        fn += len(t_list_set - p_list_k)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def macro_f1(target_lists: list, pred_lists: list, k: int):
    """Compute macro-averaged F1 at rank ``k`` across queries."""
    f1s = []
    for t_list, p_list in zip(target_lists, pred_lists):
        p_list_k = set(p_list[:k])
        t_list_set = set(t_list)
        tp = len(p_list_k & t_list_set)
        fp = len(p_list_k - t_list_set)
        fn = len(t_list_set - p_list_k)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0
