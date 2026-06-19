"""Project LeCaRDv2 embeddings through a trained retrieval encoder."""

from pathlib import Path
from typing import Tuple

import torch
from loguru import logger

from utils.model import Encoder, MultiHeadEncoder


def load_embedding_model(model_name: str, device: str) -> Encoder | MultiHeadEncoder:
    """Load an Encoder or MultiHeadEncoder checkpoint on the target device."""
    state_dict = torch.load(Path(model_name), map_location=device)

    if any(k.startswith("heads.") for k in state_dict.keys()):
        encoder = MultiHeadEncoder(dim=state_dict["heads.0.weight"].shape[1])
    else:
        encoder = Encoder(dim=state_dict["linear.weight"].shape[1])

    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    logger.info(f"Loaded {type(encoder).__name__} from {model_name} to {device}")
    return encoder


def transform_embeddings(
    model: Encoder | MultiHeadEncoder,
    embeddings: dict[int, torch.Tensor],
    device: str,
    batch_size: int = 64,
) -> dict[int, torch.Tensor]:
    """Transform an ID-indexed embedding collection in batches."""
    transformed_embeddings = {}
    id_list = list(embeddings.keys())
    for i in range(0, len(id_list), batch_size):
        batch_pids = id_list[i : i + batch_size]
        batch_embeddings = torch.stack([embeddings[pid] for pid in batch_pids]).to(device)
        with torch.no_grad():
            transformed_batch = model(batch_embeddings).cpu()
        for j, pid in enumerate(batch_pids):
            transformed_embeddings[pid] = transformed_batch[j]
    return transformed_embeddings


def load_embeddings(embedding_path: str) -> Tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Load query and candidate embeddings from a PyTorch checkpoint."""
    embedding_dict = torch.load(embedding_path)
    return embedding_dict["query_embeddings"], embedding_dict["candidate_embeddings"]


def save_embeddings(
    query_embeddings: dict[int, torch.Tensor],
    candidate_embeddings: dict[int, torch.Tensor],
    save_path: str,
) -> None:
    """Save transformed query and candidate embeddings."""
    embedding_dict = {
        "query_embeddings": query_embeddings,
        "candidate_embeddings": candidate_embeddings,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding_dict, save_path)
    logger.info(f"Saved transformed embeddings to {save_path}")


def project(model_path: str, embedding_path: str, save_path: str, device: str) -> None:
    """Apply a trained encoder to query and candidate embeddings and save them."""
    model = load_embedding_model(model_path, device)
    query_embeddings, candidate_embeddings = load_embeddings(embedding_path)
    transformed_query_embeddings = transform_embeddings(model, query_embeddings, device)
    transformed_candidate_embeddings = transform_embeddings(model, candidate_embeddings, device)
    save_embeddings(transformed_query_embeddings, transformed_candidate_embeddings, save_path)
