# KGQPP

This repository supports **query-level performance modeling** for Information Retrieval.
The current scope focuses on **query evaluation signals** derived from BM25 retrieval on
**MS MARCO Passage V1**, which serve as supervision for later representation learning
and graph-based query performance prediction.

At this stage, the repository **does not implement retrieval itself**.
Instead, it assumes the availability of query-level effectiveness scores and defines
the dataset format used by subsequent phases of the framework.

---

## Dataset

### MS MARCO Passage V1

We use **MS MARCO Passage V1** as the base benchmark.

Download the document collection:

    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz

Official documentation:
https://microsoft.github.io/msmarco/TREC-Deep-Learning.html

**Dataset statistics (V1):**
- **Queries:** 808,731 unique queries
- **Documents:** 8,841,823 passages
- **Relevance Judgments:** 532,761 labeled query–document pairs

---

## Query-Level Evaluation Signals

This project operates **exclusively at the query level**.

For each query, we assume the availability of **retrieval effectiveness scores**
(e.g., nDCG@10, MAP) obtained from a **BM25 baseline**.
These scores are treated as **query performance supervision** and are not computed
within this repository.

In other words:
- Retrieval and evaluation are performed **externally**
- This repository consumes **query-level performance signals** as input

---
## Data Setup

This repository does not include datasets.

Before running the pipeline, you must prepare:
- query files (MS MARCO V1, TREC)
- per-query BM25 evaluation files

The expected directory structure and file formats are documented in:

    dataset/README.md

You can verify your local data setup by running:

    scripts/check_data_layout.sh

---

## Project Structure

The repository is organized into the following conceptual stages:

- **Dataset & Query Evaluation**
  - Definition of query-level supervision
  - Dataset format used throughout the project

- **Phase 1: Performance Alignment**
  - Learning query representations aligned with retrieval effectiveness

- **Phase 2: Graph-based Query Performance Prediction**
  - Modeling query interactions using graph neural networks

Documentation for Phase 1 and Phase 2 will be added incrementally.



## Phase 1: Baseline Query Performance Prediction (No Fine-Tuning)

Phase 1 implements the **baseline query performance prediction pipeline** without
any performance-aligned fine-tuning of the encoder.

The goal of this phase is to:
- construct a query-centric dataset using BM25 supervision
- model query–query relationships based on semantic similarity
- train a graph neural network (GNN) to predict query performance

All representations are obtained from a **pre-trained bi-encoder**, used as-is.

---

### Inputs

Phase 1 assumes the availability of the following inputs:

- Query text files (MS MARCO V1, TREC splits)
- Per-query BM25 evaluation scores (e.g., nDCG, MAP)
- A pre-trained sentence encoder (e.g., `msmarco-distilbert-base-v4`)

Details about file locations and formats are provided in:

    dataset/README.md

---

### Pipeline Overview

Phase 1 consists of the following steps:

1. **Query–Query Similarity Mining**  
   Each query is encoded using a pre-trained bi-encoder.
   For every query, its top-K nearest neighbor queries are retrieved
   based on embedding similarity.

2. **MotherDataset Construction**  
   A unified JSON dataset is built by combining:
   - query text
   - query-level BM25 evaluation scores
   - nearest-neighbor query information

3. **Graph Construction**  
   Each query is treated as a node.
   Edges connect semantically similar queries and may be weighted by
   performance-related signals.

4. **GNN Training**  
   A graph neural network (e.g., GAT, GCN, GraphSAGE) is trained to
   regress query performance metrics such as nDCG.

5. **Evaluation**  
   Predictions are evaluated using rank correlation metrics
   (e.g., Pearson, Spearman, Kendall).

---

### Running Phase 1

To execute the full baseline pipeline, run:

    scripts/qpp_run.sh

Before running, it is recommended to verify the dataset layout:

    scripts/check_data_layout.sh

---

### Notes

- This phase uses **no encoder fine-tuning**.
- All query embeddings come from an off-the-shelf bi-encoder.
- Phase 1 serves as the baseline for later extensions.

Performance-aligned fine-tuning of the encoder is introduced
as an optional extension in Phase 2.
