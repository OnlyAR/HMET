"""Generate original-text and structured-fact embeddings for LeCaRDv2."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import click
import torch
from loguru import logger
from tqdm import tqdm
from vllm import LLM

root_path = Path(__file__).parent.parent.parent
data_path = root_path / "data" / "LeCaRDv2"
fact_data_path = data_path / "LLM_extract"

MODEL_PATHS = {
    "qwen3-embedding-0.6b": "path/to/Qwen/Qwen3-Embedding-0.6B",
}


def load_original_data() -> Tuple[Dict[int, str], Dict[int, str]]:
    query_path = data_path / "queries.jsonl"
    candidate_path = data_path / "candidates.jsonl"
    queries: List[Dict] = [json.loads(line) for line in query_path.open(encoding="utf-8")]
    candidates: List[Dict] = [json.loads(line) for line in candidate_path.open(encoding="utf-8")]
    query_data = {query["id"]: query["fact"] for query in queries}
    candidate_data = {candidate["id"]: candidate["fact"] for candidate in candidates}
    return query_data, candidate_data


def load_fact_data(path: Path) -> Dict[str, Dict[int, str]]:
    fact_data = {
        "subject": {},
        "subjective_element": {},
        "object": {},
        "objective_element": {},
    }
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            item_id = int(item["id"])
            for view in fact_data:
                fact_data[view][item_id] = item[view]
    logger.info(f"Loaded {len(fact_data['subject'])} structured facts from {path}")
    return fact_data


def get_llm(model_name: str) -> LLM:
    model_path = MODEL_PATHS[model_name]
    logger.info(f"Loading model: {model_path}")
    return LLM(
        model=model_path,
        task="embed",
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        trust_remote_code=True,
    )


def truncate(tokenizer, text: str, max_len: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_len:
        tokens = tokens[: max_len // 2] + tokens[-max_len // 2:]
    return tokenizer.decode(tokens)


def generate_embeddings(
        model: LLM,
        data: Dict[int, str],
        max_len: int,
        tokenizer=None,
) -> Dict[int, torch.Tensor]:
    logger.info("Generating text embeddings...")
    keys = []
    values = []
    for item_id, text in tqdm(data.items(), total=len(data), desc="Preparing text"):
        keys.append(item_id)
        values.append(truncate(tokenizer, text, max_len) if tokenizer is not None else text)

    outputs = model.embed(values, truncate_prompt_tokens=max_len)
    embeddings = {item_id: torch.tensor(output.outputs.embedding) for item_id, output in zip(keys, outputs)}
    logger.info("Finished generating text embeddings.")
    return embeddings


def generate_original_embeddings(model: LLM, model_name: str) -> None:
    query_data, candidate_data = load_original_data()
    tokenizer = model.get_tokenizer()
    query_embeddings = generate_embeddings(model, query_data, max_len=8192, tokenizer=tokenizer)
    candidate_embeddings = generate_embeddings(model, candidate_data, max_len=8192, tokenizer=tokenizer)

    output_dir = Path("embeddings") / "origin"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"embeddings_{model_name}.pt"
    torch.save(
        {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings,
        },
        output_path,
    )
    logger.info(f"Saved original-text embeddings to: {output_path}")


def generate_fact_embeddings(model: LLM, model_name: str) -> None:
    query_fact_data = load_fact_data(fact_data_path / "query_fact.jsonl")
    candidate_fact_data = load_fact_data(fact_data_path / "candidate_fact.jsonl")

    for view in query_fact_data:
        query_embeddings = generate_embeddings(model, query_fact_data[view], max_len=16384)
        candidate_embeddings = generate_embeddings(model, candidate_fact_data[view], max_len=16384)

        output_dir = Path("embeddings") / "fact" / view
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"embeddings_{model_name}.pt"
        torch.save(
            {
                "query_embeddings": query_embeddings,
                "candidate_embeddings": candidate_embeddings,
            },
            output_path,
        )
        logger.info(f"Saved {view} embeddings to: {output_path}")


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(list(MODEL_PATHS)),
    required=True,
    help="Embedding model name",
)
def fact2embeddings(model_name: str) -> None:
    model = get_llm(model_name)
    generate_original_embeddings(model, model_name)
    generate_fact_embeddings(model, model_name)


if __name__ == "__main__":
    fact2embeddings()
