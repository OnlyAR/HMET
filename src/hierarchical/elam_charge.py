"""Charge-clustering datasets and supervised contrastive loss for ELAM."""

import random
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ChargeClusterDataset(Dataset):
    """Group embeddings by charge for clustering and contrastive training."""

    def __init__(
        self,
        emb: Dict[int, torch.Tensor],  # ID -> Embedding (shape: [D])
        charge: Dict[int, str],  # ID -> Charge Name
    ):
        """Initialize embeddings and build a charge-to-case index."""
        self.emb = emb
        self.charge = charge
        self.ids = list(emb.keys())

        # Build an inverted index from charge names to case IDs.
        self.charge_to_ids = defaultdict(list)
        for idx in self.ids:
            c = self.charge[idx]
            self.charge_to_ids[c].append(idx)

        self.unique_charges = list(self.charge_to_ids.keys())

    def __len__(self):
        """Return the number of cases."""
        return len(self.ids)

    def __getitem__(self, index):
        """Return an embedding and its integer charge label."""
        query_id = self.ids[index]
        query_emb = self.emb[query_id]
        query_charge = self.charge[query_id]

        # Convert the charge name to a stable integer label.
        charge_label = self.unique_charges.index(query_charge)

        return {"id": query_id, "emb": query_emb, "charge_label": charge_label, "charge_name": query_charge}

    def get_triplet(self, index):
        """Sample an anchor-positive-negative triplet."""
        anchor_id = self.ids[index]
        anchor_charge = self.charge[anchor_id]

        # Select a different case with the same charge when possible.
        pos_candidates = [i for i in self.charge_to_ids[anchor_charge] if i != anchor_id]
        pos_id = random.choice(pos_candidates) if pos_candidates else anchor_id

        # Select a case with a different charge.
        neg_charge = random.choice([c for c in self.unique_charges if c != anchor_charge])
        neg_id = random.choice(self.charge_to_ids[neg_charge])

        return {"anchor": self.emb[anchor_id], "positive": self.emb[pos_id], "negative": self.emb[neg_id]}


def contrastive_collate_fn(batch):
    """Stack charge-clustering samples into a batch."""
    ids = [item["id"] for item in batch]
    embs = torch.stack([item["emb"] for item in batch])
    labels = torch.tensor([item["charge_label"] for item in batch], dtype=torch.long)

    return {"ids": ids, "embs": embs, "labels": labels}


class ChargeSupConLoss(nn.Module):
    """Supervised contrastive loss using charge labels as positives."""

    def __init__(self, temperature=0.07):
        """Initialize the contrastive temperature."""
        super(ChargeSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute charge-supervised contrastive loss.

        Args:
            features: Embeddings with shape ``(B, D)``.
            labels: Integer charge labels with shape ``(B,)``.
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features for cosine similarity.
        features = F.normalize(features, p=2, dim=1)

        # Compute the pairwise similarity matrix.
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        # Mark samples sharing the same charge.
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Exclude self-similarity along the diagonal.
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Compute stable log probabilities.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Average over samples with same-charge positives.
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))

        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_pos_pairs

        loss = mean_log_prob_pos.mean()
        return loss
