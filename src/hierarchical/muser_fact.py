"""View-aware negative sampling and distillation loss for MUSER."""

import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MuserViewAwareLegalDataset(Dataset):
    """Pair raw and teacher-view embeddings with hard-first negatives."""

    def __init__(
        self,
        q_emb: Dict[int, torch.Tensor],
        c_emb: Dict[int, torch.Tensor],
        cand_dict: Dict[int, List[int]],
        label_dict: Dict[int, Dict[int, int]],
        q_view_emb: Dict[int, torch.Tensor],
        c_view_emb: Dict[int, torch.Tensor],
        num_negative_samples: int = 16,
    ):
        """Initialize student and teacher embeddings with candidate labels."""
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.q_view_emb = q_view_emb
        self.c_view_emb = c_view_emb
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
        """Return student and teacher embeddings with sampled negatives."""
        q_id, pos_id = self.flattened_pairs[idx]

        q_emb = self.q_emb[q_id]
        pos_emb = self.c_emb[pos_id]
        q_view_emb = self.q_view_emb[q_id]
        pos_view_emb = self.c_view_emb[pos_id]

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
                # Sample with replacement when necessary.
                selected_neg_ids.extend(random.choices(easy_negs, k=needed))
            else:
                # Fall back to the positive when no negatives exist.
                selected_neg_ids.extend([pos_id] * needed)

        # Stack student and teacher negative embeddings.
        neg_embs_list = [self.c_emb[nid] for nid in selected_neg_ids]
        neg_embs = torch.stack(neg_embs_list)
        neg_view_embs_list = [self.c_view_emb[nid] for nid in selected_neg_ids]
        neg_view_embs = torch.stack(neg_view_embs_list)

        return {
            "query_emb": q_emb,  # (D,)
            "pos_emb": pos_emb,  # (D,)
            "neg_embs": neg_embs,  # (K, D)
            "query_view_emb": q_view_emb,  # (D,)
            "pos_view_emb": pos_view_emb,  # (D,)
            "neg_view_embs": neg_view_embs,  # (K, D)
            "query_id": q_id,
            "pos_id": pos_id,
            "neg_ids": selected_neg_ids,
            "positive_cids": self.query_to_positives[q_id],
        }


def view_contrastive_collate_fn(batch):
    """Collate view-aware MUSER samples into batch tensors."""

    # Stack raw student embeddings.
    query_embs = torch.stack([item["query_emb"] for item in batch])
    pos_embs = torch.stack([item["pos_emb"] for item in batch])
    neg_embs = torch.stack([item["neg_embs"] for item in batch])  # (B, K, D)

    # Stack teacher-view embeddings.
    query_view_embs = torch.stack([item["query_view_emb"] for item in batch])
    pos_view_embs = torch.stack([item["pos_view_emb"] for item in batch])
    neg_view_embs = torch.stack([item["neg_view_embs"] for item in batch])  # (B, K, D)

    return {
        "query_embs": query_embs,  # (B, D)
        "pos_embs": pos_embs,  # (B, D)
        "neg_embs": neg_embs,  # (B, K, D)
        "query_view_embs": query_view_embs,  # (B, D)
        "pos_view_embs": pos_view_embs,  # (B, D)
        "neg_view_embs": neg_view_embs,  # (B, K, D)
    }


class ViewAwareInfoNCELoss(nn.Module):
    """Combine teacher distillation, denoising, and InfoNCE learning."""

    def __init__(self, temperature=0.07, teacher_threshold=0.85, distill_weight=0.5):
        """
        Args:
            temperature: InfoNCE temperature.
            teacher_threshold: Similarity threshold for false-negative filtering.
            distill_weight: Distillation contribution to the total loss.
        """
        super().__init__()
        self.temperature = temperature
        self.teacher_threshold = teacher_threshold
        self.distill_weight = distill_weight

        # Fit student vectors to teacher vectors with MSE.
        self.mse_loss = nn.MSELoss()
        # Treat contrastive learning as candidate classification.
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_emb, pos_emb, neg_embs, query_view_emb, pos_view_emb, neg_view_embs):
        """Compute view-aware distillation and contrastive losses.

        Args:
            query_emb: Student query embeddings with shape ``(B, D)``.
            pos_emb: Student positive embeddings with shape ``(B, D)``.
            neg_embs: Student negative embeddings with shape ``(B, K, D)``.
            query_view_emb: Teacher query embeddings with shape ``(B, D)``.
            pos_view_emb: Teacher positive embeddings with shape ``(B, D)``.
            neg_view_embs: Teacher negative embeddings with shape ``(B, K, D)``.
        """

        B, K, D = neg_embs.shape

        # ===========================
        # Part 1: distillation loss.
        # ===========================
        loss_distill_q = self.mse_loss(query_emb, query_view_emb)
        loss_distill_p = self.mse_loss(pos_emb, pos_view_emb)

        # Scale MSE to a magnitude comparable to cross entropy.
        loss_distill = (loss_distill_q + loss_distill_p) / 2 * 1000

        # ===========================
        # Part 2: teacher-guided false-negative detection.
        # ===========================
        # Normalize teacher embeddings.
        t_q_norm = F.normalize(query_view_emb, p=2, dim=-1)  # (B, D)
        t_n_norm = F.normalize(neg_view_embs, p=2, dim=-1)  # (B, K, D)

        # Compute teacher query-negative similarities.
        # (B, 1, D) @ (B, D, K) -> (B, 1, K) -> (B, K)
        with torch.no_grad():
            teacher_neg_scores = torch.bmm(t_q_norm.unsqueeze(1), t_n_norm.transpose(1, 2)).squeeze(1)

            # Mark highly similar sampled negatives as false negatives.
            # (B, K)
            false_negative_mask = teacher_neg_scores > self.teacher_threshold

        # ===========================
        # Part 3: InfoNCE Loss (Student)
        # ===========================
        # Normalize student embeddings.
        s_q_norm = F.normalize(query_emb, p=2, dim=-1)  # (B, D)
        s_p_norm = F.normalize(pos_emb, p=2, dim=-1)  # (B, D)
        s_n_norm = F.normalize(neg_embs, p=2, dim=-1)  # (B, K, D)

        # Compute positive scores.
        pos_scores = (s_q_norm * s_p_norm).sum(dim=-1, keepdim=True)

        # Compute negative scores.
        neg_scores = torch.bmm(s_q_norm.unsqueeze(1), s_n_norm.transpose(1, 2)).squeeze(1)

        # Mask false negatives so they receive zero softmax probability.
        if false_negative_mask.any():
            neg_scores = neg_scores.masked_fill(false_negative_mask, -1e9)

        # Place the positive score in column zero.
        logits = torch.cat([pos_scores, neg_scores], dim=1)

        # Apply temperature scaling.
        logits = logits / self.temperature

        # The positive target is always column zero.
        targets = torch.zeros(B, dtype=torch.long, device=logits.device)
        loss_contrast = self.cross_entropy(logits, targets)

        # ===========================
        # Part 4: combine distillation and contrastive losses.
        # ===========================
        total_loss = self.distill_weight * loss_distill + (1 - self.distill_weight) * loss_contrast

        return total_loss, loss_distill, loss_contrast
