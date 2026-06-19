"""Train and evaluate hierarchical retrieval models on the LeCaRDv2 dataset."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from dense_retrieve import faiss_retrieve
from hierarchical.baseline import ContrastiveBaselineLegalDataset, SupConLoss, contrastive_collate_fn
from hierarchical.fact import ViewAwareLegalDataset, ViewAwareSupConLoss, view_contrastive_collate_fn
from loaders import load_cases, load_labels, data_path
from utils.metric import recall_at_1, recall_at_5, recall_at_10, recall_at_20
from utils.model import Encoder, MultiHeadEncoder

train_query_path = data_path / "train_queries.jsonl"
valid_query_path = data_path / "valid_queries.jsonl"
candidate_path = data_path / "candidates.jsonl"
labels = load_labels()

# Fix random seeds to ensure reproducible experiments.
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def save_model(encoder: Encoder | MultiHeadEncoder, path: Path):
    torch.save(encoder.state_dict(), path)
    logger.info(f"Saved model to {path}")


def evaluate(
        encoder: Encoder | MultiHeadEncoder,
        q_embeddings: Dict[int, torch.Tensor],
        c_embeddings: Dict[int, torch.Tensor],
        model_name: str,
        labels: Dict[int, List[int]],
        device: torch.device,
        topk: int,
):
    qids = sorted(q_embeddings.keys())
    cids = sorted(c_embeddings.keys())
    q_embeddings_list = [q_embeddings[qid] for qid in qids]
    c_embeddings_list = [c_embeddings[cid] for cid in cids]
    q_embeddings_tensor = torch.stack(q_embeddings_list)
    c_embeddings_tensor = torch.stack(c_embeddings_list)

    q_out_embeddings = encoder(q_embeddings_tensor.to(device)).cpu()
    c_out_embeddings = encoder(c_embeddings_tensor.to(device)).cpu()

    baseline_q_embeddings = {qid: emb for qid, emb in zip(qids, q_out_embeddings)}
    baseline_c_embeddings = {cid: emb for cid, emb in zip(cids, c_out_embeddings)}

    results = faiss_retrieve(baseline_q_embeddings, baseline_c_embeddings, model_name)

    final_results = {qid: items[:topk] for qid, items in results.items()}

    q_keys = sorted(final_results.keys())
    target_lists = [labels.get(qid, []) for qid in q_keys]
    pred_lists = [[item["pid"] for item in final_results[qid]] for qid in q_keys]

    return {
        "Recall@1": recall_at_1(target_lists, pred_lists),
        "Recall@5": recall_at_5(target_lists, pred_lists),
        "Recall@10": recall_at_10(target_lists, pred_lists),
        "Recall@20": recall_at_20(target_lists, pred_lists),
    }


@click.command()
@click.option("--model", type=click.Choice(["qwen3-embedding-0.6b"]), help="Model name")
@click.option("--method", type=click.Choice(["baseline", "charge", "fact", "multihead"]), help="Training method")
@click.option("--view", type=str, required=False, default=None, help="View name")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option(
    "--teacher-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Teacher similarity threshold used to identify missing labels",
)
@click.option(
    "--distill-weight",
    type=float,
    default=None,
    help="Weight of the distillation loss in the total loss (0.0-1.0)",
)
@click.option("--batch-size", type=int, default=256, show_default=True, help="Batch size")
@click.option("--temperature", type=float, default=0.15, show_default=True, help="InfoNCE temperature")
@click.option("--num-epochs", type=int, default=200, show_default=True, help="Maximum number of training epochs")
@click.option(
    "--topk",
    type=click.IntRange(min=1),
    default=200,
    show_default=True,
    help="Number of retrieved candidates used for evaluation",
)
@click.option("--out", type=str, default=None, help="Model output directory")
def train(
        model: str,
        method: str,
        view: str | None = None,
        lr: float = 1e-4,
        teacher_threshold: float = 0.85,
        distill_weight: float | None = None,
        batch_size: int = 256,
        temperature: float = 0.15,
        num_epochs: int = 200,
        topk: int = 200,
        out: str | None = None,
):
    dim = {"qwen3-embedding-0.6b": 1024}[model]

    if method == "baseline" or method == "charge":
        assert view is None, "The baseline/charge methods do not accept the view option"
        assert distill_weight is None, "The baseline/charge methods do not accept the distill_weight option"
    if method == "fact":
        assert view is not None, "The fact method requires the view option"
        assert teacher_threshold is not None, "The fact method requires the teacher_threshold option"
        assert distill_weight is not None, "The fact method requires the distill_weight option"
    if method == "multihead":
        assert view is None, "The multihead method does not accept the view option"
        assert distill_weight is None, "The multihead method does not accept the distill_weight option"

    model_output_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if method == "fact":
        model_output_filename = f"view_{teacher_threshold}_" + model_output_filename
        model_output_filename = f"distill_{distill_weight}_" + model_output_filename
    if out is not None:
        model_output_dir = out

    if method == "baseline" or method == "charge":
        model_output_dir = Path("models") / "hierarchical" / method / model / model_output_filename
    elif method == "fact":
        assert view is not None
        model_output_dir = Path("models") / "hierarchical" / method / view / model / model_output_filename
    elif method == "multihead":
        model_output_dir = Path("models") / "hierarchical" / method / model / model_output_filename
    else:
        assert False, f"Unknown training method: {method}"

    log_file = model_output_dir / "training.log"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", encoding="utf-8")
    args = {
        "model": model,
        "method": method,
        "view": view,
        "lr": lr,
        "teacher_threshold": teacher_threshold,
        "distill_weight": distill_weight,
        "batch_size": batch_size,
        "temperature": temperature,
        "num_epochs": num_epochs,
        "topk": topk,
        "out": out,
    }
    logger.info(f"All args: {args}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ----------------------------
    # Load embeddings and charge data.
    # ----------------------------
    logger.info("Loading embeddings and charge data...")
    embedding_path = Path("embeddings") / "origin" / f"embeddings_{model}.pt"

    if method == "fact":
        assert view is not None
        view_embedding_path = Path("embeddings") / "fact" / view / f"embeddings_{model}.pt"
        all_view_emb_data = torch.load(view_embedding_path)
        q_view_emb = all_view_emb_data["query_embeddings"]
        c_view_emb = all_view_emb_data["candidate_embeddings"]
    else:
        q_view_emb = None
        c_view_emb = None

    data = torch.load(embedding_path)
    q_emb = data["query_embeddings"]
    c_emb = data["candidate_embeddings"]

    train_queries = load_cases(train_query_path)
    valid_queries = load_cases(valid_query_path)
    candidates = load_cases(candidate_path)

    train_set = set(q["id"] for q in train_queries)
    valid_set = set(q["id"] for q in valid_queries)

    train_q_emb = {q: q_emb[q] for q in train_set}
    valid_q_emb = {q: q_emb[q] for q in valid_set}
    c_emb = {c["id"]: c_emb[c["id"]] for c in candidates}

    # ----------------------------
    # Dataset and DataLoader
    # ----------------------------
    logger.info("Building the dataset and DataLoader...")
    if method == "baseline" or method == "multihead":
        dataset = ContrastiveBaselineLegalDataset(
            q_emb=train_q_emb,
            c_emb=c_emb,
            labels=labels,
        )
        collate_fn = contrastive_collate_fn
    elif method == "fact":
        assert q_view_emb is not None and c_view_emb is not None
        dataset = ViewAwareLegalDataset(
            q_emb=train_q_emb,
            c_emb=c_emb,
            q_view_emb=q_view_emb,
            c_view_emb=c_view_emb,
            labels=labels,
        )
        collate_fn = view_contrastive_collate_fn
    else:
        assert False, f"Unknown training method: {method}"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ----------------------------
    # Model, loss function, and optimizer
    # ----------------------------
    logger.info("Initializing the model, loss function, and optimizer...")
    if method == "multihead":
        encoder = MultiHeadEncoder(dim=dim).to(device)
    else:
        encoder = Encoder(dim=dim).to(device)
    if method == "baseline" or method == "multihead":
        criterion = SupConLoss(temperature=temperature).to(device)
    elif method == "fact":
        assert teacher_threshold is not None and distill_weight is not None
        criterion = ViewAwareSupConLoss(
            temperature=temperature,
            teacher_threshold=teacher_threshold,
            distill_weight=distill_weight,
        ).to(device)
    else:
        assert False, f"Unknown training method: {method}"

    optimizer = optim.AdamW(encoder.parameters(), lr=lr)
    loss_list = []
    recalls = {}
    encoder.eval()
    with torch.no_grad():
        eval_metrics = evaluate(
            encoder=encoder,
            q_embeddings=valid_q_emb,
            c_embeddings=c_emb,
            model_name=model,
            labels=labels,
            device=device,
            topk=topk,
        )
    for k, v in eval_metrics.items():
        if k not in recalls:
            recalls[k] = []
        recalls[k].append(v)
    best_metric = sum(eval_metrics.values())
    best_metric_step = 0
    logger.info(f"Initial Validation Metrics: {eval_metrics}")
    save_path = model_output_dir / "encoder_epoch_best.pt"
    save_model(encoder, save_path)

    # ----------------------------
    # Training loop
    # ----------------------------

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0.0
        total_steps = 0
        for step, batch in enumerate(dataloader):
            if method == "baseline" or method == "multihead":
                query_embs = batch["query_embs"].to(device)
                pos_embs = batch["pos_embs"].to(device)
                pos_mask = batch["pos_mask"].to(device)
                neg_mask = batch["neg_mask"].to(device)

                # Forward pass
                query_proj = encoder(query_embs)
                pos_proj = encoder(pos_embs)

                loss = criterion(
                    query_proj,
                    pos_proj,
                    pos_mask,
                    neg_mask,
                )
            elif method == "fact":
                query_embs = batch["query_embs"].to(device)
                pos_embs = batch["pos_embs"].to(device)
                query_view_embs = batch["query_view_embs"].to(device)
                pos_view_embs = batch["pos_view_embs"].to(device)
                pos_mask = batch["pos_mask"].to(device)
                neg_mask = batch["neg_mask"].to(device)

                # Forward pass
                query_proj = encoder(query_embs)
                pos_proj = encoder(pos_embs)

                loss, loss_dist, loss_con = criterion(
                    student_q=query_proj,
                    student_pos=pos_proj,
                    teacher_q=query_view_embs,
                    teacher_pos=pos_view_embs,
                    pos_mask=pos_mask,
                    neg_mask=neg_mask,
                )
                logger.debug(
                    f"Step {step + 1}, Total Loss: {loss.item():.4f}, "
                    f"Distill Loss: {loss_dist.item():.4f}, "
                    f"Contrastive Loss: {loss_con.item():.4f}"
                )
            else:
                assert False, f"Unknown training method: {method}"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

        encoder.eval()
        loss_list.append(total_loss / total_steps)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / total_steps:.4f}")
        with torch.no_grad():
            eval_metrics = evaluate(
                encoder=encoder,
                q_embeddings=valid_q_emb,
                c_embeddings=c_emb,
                model_name=model,
                labels=labels,
                device=device,
                topk=topk,
            )
        for k, v in eval_metrics.items():
            if k not in recalls:
                recalls[k] = []
            recalls[k].append(v)
        logger.info(f"Epoch {epoch + 1} Validation Metrics: {eval_metrics}")
        current_sum_recall = sum(eval_metrics.values())
        if current_sum_recall > best_metric:
            best_metric = current_sum_recall
            best_metric_step = epoch + 1
            logger.info(f"New best validation sum recall {best_metric:.4f} at epoch {best_metric_step}")
            save_model(encoder, save_path)
            logger.info(f"Saved best model to {save_path}")

    # Plot the loss and recall curves.
    epochs = list(range(1, num_epochs + 1))
    rows = len(recalls) // 2 + 1
    plt.figure(figsize=(12, 12))
    plt.subplot(rows, 2, 1)
    plt.plot(epochs, loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(rows, 2, 2)
    for k, v in recalls.items():
        plt.plot([0] + epochs, v, label=k)
    # Place the legend outside the plot to avoid obscuring data.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True)
    for idx, k in enumerate(recalls.keys()):
        plt.subplot(rows, 2, idx + 3)
        plt.plot([0] + epochs, recalls[k], label=k)
        plt.title(f"{k} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.grid(True)
    plt.tight_layout()
    plot_path = model_output_dir / "training_curves.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training curves to {plot_path}")

    logger.info(f"Training complete. Best validation sum recall {best_metric:.4f} at epoch {best_metric_step}")


if __name__ == "__main__":
    train()
