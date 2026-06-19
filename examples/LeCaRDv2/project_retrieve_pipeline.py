from pathlib import Path

import torch
from embedding_project import project
from dense_retrieve import dense_retrieve, save_results
import click
from hierarchical_pipeline import metric_single

@click.command()
@click.option("--model-path", type=str, required=True, help="Path to the embedding transformation model.")
def main(model_path: str):
    model_path_obj = Path(model_path)
    model_name = model_path_obj.parent.parent.name
    method_name = model_path_obj.parent.parent.parent.name

    original_embedding_path = f"embeddings/origin/embeddings_{model_name}.pt"
    projected_embedding_path = f"embeddings/{method_name}/test_embeddings_{model_name}.pt"
    project(
        model_path=model_path,
        embedding_path=original_embedding_path,
        save_path=projected_embedding_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    results = dense_retrieve(embedding_path=projected_embedding_path)
    result_path = save_results(results, embedding_path=projected_embedding_path, debug=True)
    results = metric_single(path=result_path)
    line = "\t".join([f"{v:.4f}" for k, v in results.items()])
    with open(model_path_obj.parent / "retrieve_results.txt", "w", encoding="utf-8") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    main()