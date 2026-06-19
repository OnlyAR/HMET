"""Datasets, collation, and supervised contrastive loss for legal retrieval."""

import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ContrastiveBaselineLegalDataset(Dataset):
    """Create flattened query-positive pairs for contrastive training."""

    def __init__(
        self,
        q_emb: Dict[int, torch.Tensor],
        c_emb: Dict[int, torch.Tensor],
        labels: Dict[int, List[int]],
    ):
        """Initialize the dataset from embeddings and relevance labels."""
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.labels = labels

        self.qids = list(q_emb.keys())
        self.cid_set = set(c_emb.keys())

        # Keep only labeled positives available in the candidate embeddings.
        self.query_to_positives = {
            qid: list(set(pos_cids) & self.cid_set)
            for qid, pos_cids in labels.items()
            if qid in self.q_emb and len(set(pos_cids) & self.cid_set) > 0
        }

        # Flatten each query-positive relationship into one training example.
        self.flattened_pairs = []
        for qid, pos_cids in self.query_to_positives.items():
            for cid in pos_cids:
                self.flattened_pairs.append((qid, cid))

        print(
            f"Dataset initialized with {len(self.flattened_pairs)} total pairs "
            f"across {len(self.query_to_positives)} unique queries."
        )

    def __len__(self):
        """Return the number of flattened query-positive pairs."""
        return len(self.flattened_pairs)

    def __getitem__(self, idx):
        """Return one query-positive pair and its full positive-ID set."""
        q_id, pos_c_id = self.flattened_pairs[idx]

        q_emb = self.q_emb[q_id]
        pos_c_emb = self.c_emb[pos_c_id]

        return {
            "query_emb": q_emb,
            "pos_emb": pos_c_emb,
            "query_id": q_id,
            "pos_id": pos_c_id,
            # Retain all positives to prevent false negatives within a batch.
            "positive_cids": set(self.query_to_positives[q_id]),
        }


def contrastive_collate_fn(batch):
    """Stack samples and build positive and valid-negative masks."""
    batch_size = len(batch)

    query_embs = torch.stack([item["query_emb"] for item in batch])
    pos_embs = torch.stack([item["pos_emb"] for item in batch])

    query_ids = [item["query_id"] for item in batch]
    pos_ids = [item["pos_id"] for item in batch]
    all_positive_sets = [item["positive_cids"] for item in batch]

    # Initialize positive and valid-negative masks.
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool)
    neg_mask = torch.ones((batch_size, batch_size), dtype=torch.bool)

    for i in range(batch_size):
        for j in range(batch_size):
            # Mark candidates that are known positives for this query.
            if pos_ids[j] in all_positive_sets[i]:
                pos_mask[i, j] = True
                neg_mask[i, j] = False

            # Samples from the same query must not be treated as negatives.
            if query_ids[i] == query_ids[j]:
                neg_mask[i, j] = False

    return {
        "query_embs": query_embs,  # (B, D)
        "pos_embs": pos_embs,  # (B, D)
        "pos_mask": pos_mask,  # (B, B), all known positive pairs
        "neg_mask": neg_mask,  # (B, B), valid negative pairs
    }


class SupConLoss(nn.Module):
    """Supervised contrastive loss with support for multiple positives."""

    def __init__(self, temperature=0.07):
        """Initialize the loss temperature."""
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embs, pos_embs, pos_mask, neg_mask):
        """
        Args:
            query_embs: (B, D)
            pos_embs:   (B, D)
            pos_mask: Positive-pair mask with shape ``(B, B)``.
            neg_mask: Valid-negative mask with shape ``(B, B)``.
        """
        # Normalize features before computing cosine similarities.
        query_embs = nn.functional.normalize(query_embs, dim=-1)
        pos_embs = nn.functional.normalize(pos_embs, dim=-1)

        # Compute and scale the pairwise similarity matrix.
        logits = torch.matmul(query_embs, pos_embs.T) / self.temperature

        # Exclude entries that are neither positives nor valid negatives.
        valid_mask = pos_mask | neg_mask

        # Subtract the row maximum for numerical stability.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute the denominator over valid entries only.
        exp_logits = torch.exp(logits) * valid_mask.float()
        log_prob_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute log probabilities for positive pairs.
        log_prob = (logits - log_prob_denom) * pos_mask.float()

        # Average over all positives in each valid row.
        num_pos_per_row = pos_mask.sum(1)
        # Ignore rows without positives.
        valid_rows = num_pos_per_row > 0
        row_loss = -log_prob.sum(1)[valid_rows] / num_pos_per_row[valid_rows]

        return row_loss.mean()
