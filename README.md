# Hierarchical Multi-view Embedding Transformation for Legal Case Retrieval

Official implementation of **Hierarchical Multi-view Embedding Transformation (HMET)** for legal case retrieval.

HMET improves dense legal case retrieval by extracting multiple structured legal views, using their embeddings as teacher signals to transform the original case embeddings, and combining view-specific Gaussian similarities in a hierarchical retrieval pipeline.

The repository provides data for three legal case retrieval datasets:

- **LeCaRDv2**
- **ELAM**
- **MUSER**

The runnable workflow under `examples/LeCaRDv2/` provides one concrete example of the HMET preprocessing, training, and retrieval pipeline. The repository also includes the ELAM and MUSER data used by the project; their dataset-specific experiment organization may be adapted from the LeCaRDv2 example.

## Method overview

The current implementation uses four legal-fact views:

- **Subject**: the criminal subject and its legal capacity.
- **Subjective element**: intent, negligence, and other subjective aspects.
- **Object**: the legal interest or social relationship infringed by the offense.
- **Objective element**: conduct, time, location, and consequences of the offense.

The pipeline contains four stages:

1. Extract the four structured views from each query and candidate case with an instruction-tuned LLM.
2. Generate embeddings for both the original case facts and the extracted views.
3. Train an embedding transformation model using supervised contrastive learning and view-aware distillation.
4. Use raw embeddings to retrieve the top-200 candidates, then rerank them by the weighted sum of the four view-specific Gaussian similarities.

## Repository structure

```text
HMET/
├── data/
│   ├── LeCaRDv2/                 # LeCaRDv2 data and extracted legal views
│   ├── ELAM/                     # ELAM dataset files
│   └── MUSER/                    # MUSER dataset files
├── examples/
│   └── LeCaRDv2/
│       ├── query_info_extract.py # Extract structured legal views
│       ├── fact2embeddings.py    # Generate original and view embeddings
│       ├── train.py              # Train embedding transformations
│       ├── embedding_project.py  # Apply a trained transformation
│       ├── dense_retrieve.py     # FAISS retrieval and evaluation
│       ├── project_retrieve_pipeline.py
│       ├── hierarchical_pipeline.py
│       └── loaders.py
├── src/
│   ├── hierarchical/             # Datasets and training objectives
│   └── utils/                    # Models, metrics, and retrieval utilities
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12.4
- PyTorch
- CUDA-capable GPU recommended
- vLLM 0.9.0
- FAISS

Install uv and synchronize the project environment from the repository root:

```bash
uv python install 3.12.4
uv sync
```

Run project commands through the managed environment with `uv run`, or activate it manually with `source .venv/bin/activate`.

The embedding and information-extraction stages are designed for GPU execution. Check the vLLM and PyTorch CUDA compatibility requirements for your system.

Set the module search path before running the example scripts:

```bash
export PYTHONPATH="$PWD/src:$PWD/examples/LeCaRDv2"
```

All commands below assume that they are executed from the repository root and the uv environment is active. Without activation, prefix each command with `uv run`.

## Datasets

This repository provides the following dataset directories:

| Dataset | Directory | Included example workflow |
|---|---|---|
| LeCaRDv2 | `data/LeCaRDv2/` | `examples/LeCaRDv2/` |
| ELAM | `data/ELAM/` | Use the LeCaRDv2 workflow as a reference |
| MUSER | `data/MUSER/` | Use the LeCaRDv2 workflow as a reference |

The dataset layouts currently included in the repository are:

```text
data/
├── LeCaRDv2/
│   ├── queries.jsonl
│   ├── query_allcontext.json
│   ├── candidates.jsonl
│   ├── labels.json
│   ├── train_queries.jsonl
│   ├── valid_queries.jsonl
│   ├── test_queries.jsonl
│   └── test_qids.json
├── ELAM/
│   ├── intId_sentences_dict_noh.jsonl
│   ├── labels.json
│   ├── train_set.json
│   ├── valid_set.json
│   └── test_set.json
└── MUSER/
    ├── cases.jsonl
    ├── labels.json
    ├── cands_by_query.json
    ├── top30_dict.json
    ├── train_set.json
    └── test_set.json
```

Use each dataset in accordance with its original license and terms of use. Please cite the corresponding dataset paper when reporting experimental results.

## LeCaRDv2 example pipeline

All preprocessing, training, inference, and fusion commands below apply specifically to **LeCaRDv2**.

The LeCaRDv2 files are located under `data/LeCaRDv2/`:

```text
data/LeCaRDv2/
├── queries.jsonl
├── query_allcontext.json
├── candidates.jsonl
├── labels.json
├── train_queries.jsonl
├── valid_queries.jsonl
├── test_queries.jsonl
└── test_qids.json
```

Query and candidate JSONL records must contain at least:

```json
{"id": 1, "fact": "Case description"}
```

`labels.json` maps each query ID to its relevant candidate cases.

## Model configuration

Before running preprocessing, replace the placeholder model paths in:

- `examples/LeCaRDv2/query_info_extract.py`
- `examples/LeCaRDv2/fact2embeddings.py`

For example:

```python
MODEL_PATHS = {
    "qwen3-embedding-0.6b": "/path/to/Qwen3-Embedding-0.6B",
}
```

The current implementation assumes 1,024-dimensional embeddings for `qwen3-embedding-0.6b`. When adding another embedding model, update the model choices and dimensions consistently in `train.py` and `dense_retrieve.py`.

## Preprocessing

### 1. Extract structured legal views

```bash
python examples/LeCaRDv2/query_info_extract.py
```

Expected outputs:

```text
data/LeCaRDv2/LLM_extract/
├── query_fact.jsonl
└── candidate_fact.jsonl
```

Each output record contains `subject`, `subjective_element`, `object`, and `objective_element`.

### 2. Generate embeddings

```bash
python examples/LeCaRDv2/fact2embeddings.py \
  --model-name qwen3-embedding-0.6b
```

The expected embedding layout used by the training and retrieval pipeline is:

```text
embeddings/
├── origin/
│   └── embeddings_qwen3-embedding-0.6b.pt
└── fact/
    ├── subject/
    │   └── embeddings_qwen3-embedding-0.6b.pt
    ├── subjective_element/
    │   └── embeddings_qwen3-embedding-0.6b.pt
    ├── object/
    │   └── embeddings_qwen3-embedding-0.6b.pt
    └── objective_element/
        └── embeddings_qwen3-embedding-0.6b.pt
```

Original/view embedding checkpoints must contain:

```python
{
    "query_embeddings": {query_id: tensor},
    "candidate_embeddings": {candidate_id: tensor},
}
```

## Training

### Baseline transformation

```bash
python examples/LeCaRDv2/train.py \
  --model qwen3-embedding-0.6b \
  --method baseline \
  --lr 1e-4 \
  --batch-size 256 \
  --temperature 0.15 \
  --num-epochs 200 \
  --topk 200
```

### Multi-head transformation

```bash
python examples/LeCaRDv2/train.py \
  --model qwen3-embedding-0.6b \
  --method multihead \
  --lr 1e-4 \
  --batch-size 256 \
  --temperature 0.15 \
  --num-epochs 200
```

### View-aware HMET training

Train one transformation for each legal view:

```bash
python examples/LeCaRDv2/train.py \
  --model qwen3-embedding-0.6b \
  --method fact \
  --view subject \
  --teacher-threshold 0.85 \
  --distill-weight 0.5 \
  --lr 1e-4 \
  --batch-size 256 \
  --temperature 0.15 \
  --num-epochs 200 \
  --topk 200
```

`--teacher-threshold` defaults to `0.85`. The paper settings are used as the CLI defaults for batch size (`256`), temperature (`0.15`), and maximum epochs (`200`).

Valid values for `--view` are:

```text
subject
subjective_element
object
objective_element
```

Repeat the command for all four views.

Training outputs are written under:

```text
models/hierarchical/<method>/.../
├── encoder_epoch_best.pt
├── training.log
└── training_curves.png
```

The checkpoint with the highest sum of validation Recall@1, Recall@5, Recall@10, and Recall@20 is saved as `encoder_epoch_best.pt`.

## Inference and retrieval

### Apply a trained transformation

`embedding_project.py` loads a trained `Encoder` or `MultiHeadEncoder`, transforms all query and candidate embeddings, and saves the projected embeddings. It is called by:

```bash
python examples/LeCaRDv2/project_retrieve_pipeline.py \
  --model-path models/hierarchical/baseline/qwen3-embedding-0.6b/<run>/encoder_epoch_best.pt
```

The pipeline then performs FAISS retrieval, saves ranked results under `gear_results/`, and reports retrieval metrics.

### Retrieve directly from an embedding checkpoint

```bash
python examples/LeCaRDv2/dense_retrieve.py \
  --path embeddings/fact/subject/embeddings_qwen3-embedding-0.6b.pt
```

The retrieval implementation uses an exact FAISS L2 index and reports:

- Recall@1
- Recall@5
- Recall@10
- Recall@20
- MRR

## Multi-view fusion

After generating retrieval results for the raw embeddings and four projected views, run hierarchical weighted-score fusion:

```bash
python examples/LeCaRDv2/hierarchical_pipeline.py \
  --model qwen3-embedding-0.6b \
  --subject-weight 0.25 \
  --subjective-element-weight 0.25 \
  --object-weight 0.25 \
  --objective-element-weight 0.25
```

Weights are provided manually, must be non-negative, and must sum to one. The raw embedding ranking supplies the top-200 coarse candidate set. Within that set, each view contributes

```text
weight_k * exp(-||q_k - d_k||²)
```

to the final score, matching the aggregation defined in the paper.

The fusion pipeline expects retrieval files in:

```text
gear_results/
├── origin/embeddings_qwen3-embedding-0.6b.json
└── fact/
    ├── subject/embeddings_qwen3-embedding-0.6b.json
    ├── subjective_element/embeddings_qwen3-embedding-0.6b.json
    ├── object/embeddings_qwen3-embedding-0.6b.json
    └── objective_element/embeddings_qwen3-embedding-0.6b.json
```

## Implementation notes

- The repository currently contains placeholder local model paths that must be configured manually.
- The `charge` option is present in the training CLI but is not connected to the current LeCaRDv2 training branch.
- The `--out` option in `train.py` is currently overridden by the method-specific output path construction.
- Final test metrics should only be reported after all hyperparameters, thresholds, and fusion weights have been selected on validation data.
