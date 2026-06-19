"""View-aware datasets and losses for teacher-guided legal retrieval."""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ViewAwareLegalDataset(Dataset):
    """Pair raw embeddings with view-specific teacher embeddings."""

    def __init__(
        self,
        q_emb: Dict[int, torch.Tensor],
        c_emb: Dict[int, torch.Tensor],
        q_view_emb: Dict[int, torch.Tensor],
        c_view_emb: Dict[int, torch.Tensor],
        labels: Dict[int, List[int]],
    ):
        """Initialize raw and teacher-view embeddings with relevance labels."""
        self.q_emb = q_emb
        self.c_emb = c_emb
        self.q_view_emb = q_view_emb
        self.c_view_emb = c_view_emb
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

        print(f"ViewAwareDataset initialized with {len(self.flattened_pairs)} pairs.")

    def __len__(self):
        """Return the number of flattened query-positive pairs."""
        return len(self.flattened_pairs)

    def __getitem__(self, idx):
        """Return raw and teacher-view embeddings for one positive pair."""
        q_id, pos_c_id = self.flattened_pairs[idx]

        # Raw embeddings are student inputs.
        q_raw = self.q_emb[q_id]
        pos_c_raw = self.c_emb[pos_c_id]

        # View embeddings provide teacher targets and denoising signals.
        q_view = self.q_view_emb[q_id]
        pos_c_view = self.c_view_emb[pos_c_id]

        return {
            "query_emb": q_raw,
            "pos_emb": pos_c_raw,
            "query_view_emb": q_view,
            "pos_view_emb": pos_c_view,
            "query_id": q_id,
            "pos_id": pos_c_id,
            "positive_cids": set(self.query_to_positives[q_id]),
        }


def view_contrastive_collate_fn(batch):
    """Stack view-aware samples and construct contrastive masks."""
    batch_size = len(batch)

    # Stack raw student embeddings.
    query_embs = torch.stack([item["query_emb"] for item in batch])
    pos_embs = torch.stack([item["pos_emb"] for item in batch])

    # Stack teacher-view embeddings.
    query_view_embs = torch.stack([item["query_view_emb"] for item in batch])
    pos_view_embs = torch.stack([item["pos_view_emb"] for item in batch])

    query_ids = [item["query_id"] for item in batch]
    pos_ids = [item["pos_id"] for item in batch]
    all_positive_sets = [item["positive_cids"] for item in batch]

    # Build masks from the official relevance labels.
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool)
    neg_mask = torch.ones((batch_size, batch_size), dtype=torch.bool)

    for i in range(batch_size):
        for j in range(batch_size):
            if pos_ids[j] in all_positive_sets[i]:
                pos_mask[i, j] = True
                neg_mask[i, j] = False

            if query_ids[i] == query_ids[j]:
                neg_mask[i, j] = False

    return {
        "query_embs": query_embs,  # (B, D_raw)
        "pos_embs": pos_embs,  # (B, D_raw)
        "query_view_embs": query_view_embs,  # (B, D_view)
        "pos_view_embs": pos_view_embs,  # (B, D_view)
        "pos_mask": pos_mask,  # (B, B) - Hard Positive Labels
        "neg_mask": neg_mask,  # (B, B) - Hard Negative Candidates
    }


class ViewAwareSupConLoss(nn.Module):
    """Combine teacher distillation with supervised contrastive learning."""

    def __init__(self, temperature=0.07, teacher_threshold=0.85, distill_weight=0.5):
        """
        Args:
            temperature: Contrastive temperature.
            teacher_threshold: Similarity threshold for potential false negatives.
            distill_weight: Distillation contribution to the total loss.
        """
        super().__init__()
        self.temperature = temperature
        self.teacher_threshold = teacher_threshold
        self.distill_weight = distill_weight

        # MSE preserves both teacher direction and magnitude information.
        self.distill_loss_fn = nn.MSELoss()

    def forward(self, student_q, student_pos, teacher_q, teacher_pos, pos_mask, neg_mask):
        """
        Args:
            student_q: Projected query embeddings with shape ``(B, D)``.
            student_pos: Projected candidate embeddings with shape ``(B, D)``.
            teacher_q: Teacher query embeddings with shape ``(B, D)``.
            teacher_pos: Teacher candidate embeddings with shape ``(B, D)``.
            pos_mask: Official positive-pair mask.
            neg_mask: Initial valid-negative mask.
        """

        # ===========================
        # Part 1: distillation loss.
        # ===========================
        loss_distill_q = self.distill_loss_fn(student_q, teacher_q)
        loss_distill_p = self.distill_loss_fn(student_pos, teacher_pos)
        loss_distill = (loss_distill_q + loss_distill_p) / 2 * 1000

        # ===========================
        # Part 2: teacher-guided negative-mask refinement.
        # ===========================
        # Normalize teacher embeddings for cosine similarity.
        teacher_q_norm = F.normalize(teacher_q, dim=-1)
        teacher_pos_norm = F.normalize(teacher_pos, dim=-1)

        # Compute the teacher similarity matrix.
        with torch.no_grad():
            teacher_sim = torch.matmul(teacher_q_norm, teacher_pos_norm.T)

            # Treat unlabeled but highly similar pairs as potential false negatives.
            potential_false_negatives = (teacher_sim > self.teacher_threshold) & (~pos_mask)

            # Remove potential false negatives from the contrastive denominator.
            refined_neg_mask = neg_mask & (~potential_false_negatives)

        # ===========================
        # Part 3: supervised contrastive loss.
        # ===========================
        # Normalize student embeddings.
        student_q_norm = F.normalize(student_q, dim=-1)
        student_pos_norm = F.normalize(student_pos, dim=-1)

        # Compute pairwise logits.
        logits = torch.matmul(student_q_norm, student_pos_norm.T) / self.temperature

        # Use official positives and teacher-refined negatives.
        valid_mask = pos_mask | refined_neg_mask

        # Apply the log-sum-exp stability adjustment.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute the denominator over valid entries.
        exp_logits = torch.exp(logits) * valid_mask.float()
        log_prob_denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute log probabilities.
        log_prob = logits - log_prob_denom

        # Optimize official positives only; teacher matches are ignored, not promoted.
        mask_pos_float = pos_mask.float()
        log_prob = log_prob * mask_pos_float

        # Average positive log probabilities within each valid row.
        num_pos_per_row = mask_pos_float.sum(1)
        valid_rows = num_pos_per_row > 0

        if valid_rows.sum() > 0:
            loss_contrast = -log_prob.sum(1)[valid_rows] / num_pos_per_row[valid_rows]
            loss_contrast = loss_contrast.mean()
        else:
            loss_contrast = torch.tensor(0.0, device=student_q.device, requires_grad=True)

        # ===========================
        # Part 4: combine distillation and contrastive losses.
        # ===========================
        total_loss = self.distill_weight * loss_distill + (1 - self.distill_weight) * loss_contrast

        return total_loss, loss_distill, loss_contrast
