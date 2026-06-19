"""Baseline negative-sampling dataset and InfoNCE loss for MUSER."""

import sys
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MuserBaselineLegalDataset(Dataset):
    """Build positive pairs with hard-first multi-negative sampling."""

    def __init__(
        self,
        q_emb: Dict[int, torch.Tensor],
        c_emb: Dict[int, torch.Tensor],
        cand_dict: Dict[int, List[int]],
        label_dict: Dict[int, Dict[int, int]],
        num_negative_samples: int = 16,
    ):
        """
        Args:
            q_emb: Query embeddings
            c_emb: Candidate embeddings
            cand_dict: Query ID -> List of all candidate IDs
            label_dict: Query ID -> {Candidate ID -> Score} (Top 30 scores)
            num_negative_samples: Number of negatives sampled per query.
        """
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.cid_set = set(c_emb.keys())
        self.num_negatives = num_negative_samples

        self.query_to_positives = {}
        self.query_to_hard_negs = {}
        self.query_to_easy_negs = {}
        self.flattened_pairs = []

        # Classify candidates into positive, hard-negative, and easy-negative pools.
        valid_queries = 0
        for qid, candidates in cand_dict.items():
            if qid not in self.q_emb:
                continue

            scores = label_dict.get(qid, {})
            pos_cids = []
            hard_neg_cids = []
            easy_neg_cids = []

            for cid in candidates:
                if cid not in self.cid_set:
                    continue

                # High scores are positives; low labeled scores are hard negatives.
                if cid in scores:
                    if scores[cid] >= 5:
                        pos_cids.append(cid)
                    elif scores[cid] < 3:
                        hard_neg_cids.append(cid)
                else:
                    easy_neg_cids.append(cid)

            if not pos_cids:
                continue

            valid_queries += 1

            self.query_to_positives[qid] = set(pos_cids)
            self.query_to_hard_negs[qid] = hard_neg_cids
            self.query_to_easy_negs[qid] = easy_neg_cids

            # Create one training example per positive candidate.
            for pos_cid in pos_cids:
                self.flattened_pairs.append((qid, pos_cid))

        print(
            f"Dataset initialized with {len(self.flattened_pairs)} pairs. "
            f"Negative samples per pair: {self.num_negatives}"
        )

    def __len__(self):
        """Return the number of flattened query-positive pairs."""
        return len(self.flattened_pairs)

    def __getitem__(self, idx):
        """Return one positive pair with a fixed number of sampled negatives."""
        q_id, pos_id = self.flattened_pairs[idx]

        q_emb = self.q_emb[q_id]
        pos_emb = self.c_emb[pos_id]

        # Prefer hard negatives and fill remaining slots with easy negatives.
        hard_negs = self.query_to_hard_negs[q_id]
        easy_negs = self.query_to_easy_negs[q_id]

        selected_neg_ids = []

        # First sample hard negatives.
        if len(hard_negs) > 0:
            if len(hard_negs) >= self.num_negatives:
                selected_neg_ids.extend(random.sample(hard_negs, self.num_negatives))
            else:
                selected_neg_ids.extend(hard_negs)

        # Determine how many slots remain.
        needed = self.num_negatives - len(selected_neg_ids)

        # Fill remaining slots with easy negatives.
        if needed > 0:
            if len(easy_negs) >= needed:
                selected_neg_ids.extend(random.sample(easy_negs, needed))
            elif len(easy_negs) > 0:
                # Sample with replacement when the easy-negative pool is too small.
                selected_neg_ids.extend(random.choices(easy_negs, k=needed))
            else:
                # Fall back to the positive when no negatives exist.
                selected_neg_ids.extend([pos_id] * needed)

        # Stack negative embeddings into shape (K, D).
        neg_embs_list = [self.c_emb[nid] for nid in selected_neg_ids]
        neg_embs = torch.stack(neg_embs_list)

        return {
            "query_emb": q_emb,  # (D,)
            "pos_emb": pos_emb,  # (D,)
            "neg_embs": neg_embs,  # (K, D)
            "query_id": q_id,
            "pos_id": pos_id,
            "neg_ids": selected_neg_ids,
            # Retain all positives to support false-negative filtering.
            "positive_cids": self.query_to_positives[q_id],
        }


def contrastive_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate MUSER samples with multiple negatives into batch tensors.

    Args:
        batch: Samples returned by ``MuserBaselineLegalDataset``.

    Returns:
        Batched embeddings, IDs, and positive-ID sets.
    """

    # Stack embedding tensors.
    # query_emb: [ (D,), (D,), ... ] -> (B, D)
    query_embs = torch.stack([item["query_emb"] for item in batch])

    # pos_emb: [ (D,), (D,), ... ] -> (B, D)
    pos_embs = torch.stack([item["pos_emb"] for item in batch])

    # neg_embs: [ (K, D), (K, D), ... ] -> (B, K, D)
    # Every sample contains the same number of negatives.
    neg_embs = torch.stack([item["neg_embs"] for item in batch])

    # Keep IDs as Python lists for inspection and mask construction.
    query_ids = [item["query_id"] for item in batch]
    pos_ids = [item["pos_id"] for item in batch]

    neg_ids = [item["neg_ids"] for item in batch]

    # Preserve per-query positive sets.
    positive_cids = [item["positive_cids"] for item in batch]

    return {
        "query_emb": query_embs,
        "pos_emb": pos_embs,
        "neg_embs": neg_embs,  # (B, K, D)
        "query_id": query_ids,
        "pos_id": pos_ids,
        "neg_ids": neg_ids,  # List[List[int]]
        "positive_cids": positive_cids,
    }


class InfoNCELoss(nn.Module):
    """InfoNCE loss over one positive and multiple negatives."""

    def __init__(self, temperature=0.05):
        """Initialize the loss temperature."""
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, pos_emb, neg_embs):
        """Compute InfoNCE loss for one positive and ``K`` negatives.

        Args:
            - "query_emb": (B, D)
            - "pos_emb":   (B, D)
            - ``neg_embs``: ``(B, K, D)``
        """

        B, K, D = neg_embs.shape

        # Normalize embeddings for cosine similarity.
        query = F.normalize(query_emb, p=2, dim=-1)
        pos_doc = F.normalize(pos_emb, p=2, dim=-1)
        neg_docs = F.normalize(neg_embs, p=2, dim=-1)

        # Compute positive scores.
        # (B, D) * (B, D) -> (B, 1) element-wise multiply then sum
        pos_scores = torch.sum(query * pos_doc, dim=-1, keepdim=True)  # Shape: (B, 1)

        # Compute negative scores.
        # Query: (B, 1, D)
        # Negs Transposed: (B, D, K)
        # Result: (B, 1, K) -> Squeeze -> (B, K)
        neg_scores = torch.bmm(query.unsqueeze(1), neg_docs.transpose(1, 2)).squeeze(1)

        # Place positive scores in column zero.
        # Shape: (B, 1 + K)
        logits = torch.cat([pos_scores, neg_scores], dim=1)

        # Apply temperature scaling.
        logits = logits / self.temperature

        # The positive target is always column zero.
        targets = torch.zeros(B, dtype=torch.long, device=logits.device)

        loss = self.criterion(logits, targets)

        return loss
