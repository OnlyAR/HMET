"""Charge-aware sampling and InfoNCE loss for legal case retrieval."""

import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def compute_case_similarity(query_charges, candidate_charges, score_graph, query_id: int, candidate_id: int) -> float:
    """Compute average pairwise charge similarity between two cases."""
    q_charges = query_charges.get(query_id, [])
    c_charges = candidate_charges.get(candidate_id, [])
    similarity = 0.0
    for q_charge in q_charges:
        for c_charge in c_charges:
            charge_score = score_graph[q_charge].get(c_charge, 0.0)
            similarity += charge_score
    return similarity / (len(q_charges) * len(c_charges))


class ContrastiveChargeLegalDataset(Dataset):
    """Sample charge-similar positives and charge-dissimilar negatives."""

    def __init__(
        self,
        q_emb: Dict[int, torch.Tensor],
        c_emb: Dict[int, torch.Tensor],
        q_charges: Dict[int, List[str]],
        c_charges: Dict[int, List[str]],
        score_graph: Dict[str, Dict[str, float]],
        pos_threshold: float = 0.2,
        neg_threshold: float = 0.1,
        num_negatives: int = 5,
    ):
        """Initialize embeddings, charge metadata, and sampling thresholds."""
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.q_charges = q_charges
        self.c_charges = c_charges
        self.score_graph = score_graph
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.num_negatives = num_negatives

        # Collect all query and candidate IDs.
        self.query_ids = list(q_emb.keys())  # [int]
        self.candidate_ids = list(c_emb.keys())  # [int]

        # Cache entries may be populated externally for repeated use.
        self.precomputed_similarities = {}

        # Precompute positive and negative pools for efficient sampling.
        self.query_to_positives = {}
        self.query_to_negatives = {}

        for q_id in self.query_ids:
            positives = []
            negatives = []
            for c_id in self.candidate_ids:
                # Reuse cached similarity when available.
                sim_key = (q_id, c_id)
                if sim_key in self.precomputed_similarities:
                    similarity = self.precomputed_similarities[sim_key]
                else:
                    # Compute missing similarities on demand.
                    similarity = compute_case_similarity(self.q_charges, self.c_charges, self.score_graph, q_id, c_id)

                if similarity > self.pos_threshold:
                    positives.append((c_id, similarity))
                elif similarity < self.neg_threshold:
                    negatives.append((c_id, similarity))

            self.query_to_positives[q_id] = positives
            self.query_to_negatives[q_id] = negatives

        # Retain queries that have both positive and negative candidates.
        self.valid_query_ids = [
            q_id
            for q_id in self.query_ids
            if len(self.query_to_positives[q_id]) > 0 and len(self.query_to_negatives[q_id]) > 0
        ]
        print(f"Dataset initialized with {len(self.valid_query_ids)} valid queries out of {len(self.query_ids)} total.")

    def __len__(self):
        """Return the number of valid queries."""
        return len(self.valid_query_ids)

    def __getitem__(self, idx):
        """Sample one positive and multiple negatives for a query."""
        q_id = self.valid_query_ids[idx]
        q_emb = self.q_emb[q_id]
        q_charges = self.q_charges.get(q_id, [])

        # Select one positive candidate.
        pos_candidates_with_sim = self.query_to_positives[q_id]
        pos_c_id, pos_similarity = random.choice(pos_candidates_with_sim)
        pos_c_emb = self.c_emb[pos_c_id]
        pos_c_charges = self.c_charges.get(pos_c_id, [])

        # Sample negative candidates.
        neg_candidates_with_sim = self.query_to_negatives[q_id]
        sampled_negs = random.sample(neg_candidates_with_sim, min(self.num_negatives, len(neg_candidates_with_sim)))

        neg_c_embs = [self.c_emb[n_id] for n_id, _ in sampled_negs]
        neg_c_charges_list = [self.c_charges.get(n_id, []) for n_id, _ in sampled_negs]
        neg_similarities = [sim for _, sim in sampled_negs]

        # Stack negative embeddings into one tensor.
        stacked_neg_c_embs = torch.stack(neg_c_embs)

        return {
            "query_emb": q_emb,
            "pos_emb": pos_c_emb,
            "neg_embs": stacked_neg_c_embs,  # Shape: (num_negatives, embedding_dim)
            "query_id": q_id,
            "pos_id": pos_c_id,
            "neg_ids": [n_id for n_id, _ in sampled_negs],
        }


class ChargeInfoNCELoss(nn.Module):
    """InfoNCE loss over one positive and multiple charge-based negatives."""

    def __init__(self, temperature=0.07, reduction="mean"):
        """Initialize temperature scaling and reduction behavior."""
        super(ChargeInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)  # Apply log-softmax along the candidates dimension

    def forward(self, query_emb, pos_emb, neg_embs):
        """Compute InfoNCE loss.

        Args:
            query_emb: Query embeddings with shape ``(B, D)``.
            pos_emb: Positive embeddings with shape ``(B, D)``.
            neg_embs: Negative embeddings with shape ``(B, K, D)``.

        Returns:
            Reduced or per-sample InfoNCE loss.
        """

        # Compute positive similarities.
        pos_sim = torch.sum(query_emb * pos_emb, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Compute negative similarities.
        neg_sim = torch.bmm(neg_embs, query_emb.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, num_negatives)

        # Concatenate positive and negative logits.
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # Shape: (batch_size, 1 + num_negatives)

        # Apply temperature scaling.
        logits /= self.temperature

        # Convert logits into log probabilities.
        log_probs = self.log_softmax(logits)  # Shape: (batch_size, 1 + num_negatives)

        # The positive candidate is always in column zero.
        loss = -log_probs[:, 0]  # Shape: (batch_size,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
