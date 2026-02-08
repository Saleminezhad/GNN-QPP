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

## Dataset Format

Each query is represented together with its associated performance signal.

Conceptually, the dataset follows this structure:

    Query ID | Query Text | Performance Score (e.g., nDCG@10)

This query-centric representation enables:
- performance-aware query representation learning
- construction of query–query relationships
- graph-based query performance prediction

**Note:**
Document-level retrieval results are not stored or processed in this repository.

---

## Project Structure

The repository is organized into the following conceptual stages:

- **Dataset & Query Evaluation (current scope)**
  - Definition of query-level supervision
  - Dataset format used throughout the project

- **Phase 1: Performance Alignment**
  - Learning query representations aligned with retrieval effectiveness

- **Phase 2: Graph-based Query Performance Prediction**
  - Modeling query interactions using graph neural networks

Documentation for Phase 1 and Phase 2 will be added incrementally.

---

## Scope Note

This repository intentionally excludes:
- dense retriever pipelines
- FAISS indexing
- document embeddings
- document-level graph construction

These components are not required for the **query-centric QPP framework**
developed in this project.

---

## Upcoming Additions

- Phase 1: Performance-aligned query encoder training
- Phase 2: Graph construction and GNN training
- Experimental settings and reproducibility notes