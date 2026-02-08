# Dataset Layout (Query-Level, BM25 Supervision)

This project assumes you already have:
1) MS MARCO Passage V1 query files (TSV)
2) BM25 evaluation outputs for those queries (TSV with per-query metrics)

**Important:** the dataset files are not tracked by Git. Create the following directory structure locally.

---

## Expected Folder Structure

Place data under `dataset/` as follows:

    dataset/
      v1/
        queries.train.small.tsv
      trec/
        2019/
          queries.tsv
        2020/
          queries.tsv
        hard/
          queries.tsv
      bm25_eval/
        v1/
          train_eval.tsv
          2019_eval.tsv
          2020_eval.tsv
          hard_eval.tsv

Your scripts will refer to these paths (e.g., `dataset/v1/queries.train.small.tsv`).

---

## Query Files

### 1) MS MARCO V1 training queries

Path:

    dataset/v1/queries.train.small.tsv

Format (TSV):

    <qid>\t<query_text>

Example:

    121352  what is the purpose of a constitution
    634306  symptoms of low blood sugar

### 2) TREC query sets (optional splits)

Paths:

    dataset/trec/2019/queries.tsv
    dataset/trec/2020/queries.tsv
    dataset/trec/hard/queries.tsv

Same format (TSV):

    <qid>\t<query_text>

---

## BM25 Evaluation Files (Per-Query Metrics)

These files contain per-query effectiveness scores computed from BM25 runs using Pyserini/Anserini.

Paths:

    dataset/bm25_eval/v1/train_eval.tsv
    dataset/bm25_eval/v1/2019_eval.tsv
    dataset/bm25_eval/v1/2020_eval.tsv
    dataset/bm25_eval/v1/hard_eval.tsv

Required columns (TSV header):

    qid    map    recip_rank    ndcg

Example:

    qid     map     recip_rank     ndcg
    2082    0.2335994111985414     1.0     0.5053510319011574

Notes:
- Metrics are typically computed at a cutoff such as 1000 (depending on your evaluation command).
- Only query-level scores are used by this project.

---

## How to Generate BM25 Evaluation Files (Pyserini/Anserini)

You need to:
1) run BM25 retrieval to produce a run file
2) evaluate the run against qrels
3) convert the evaluation output into a TSV like the example above

High-level workflow:

    BM25 retrieval (run)  ->  run file
    Evaluate vs qrels     ->  metric output
    Parse into TSV        ->  train_eval.tsv / 2019_eval.tsv / ...

If you already have files such as:

    dataset/Bm25/eval/eval_msmarco-v1-passage.bm25-default.<split>.txt_1000

You can convert them into the TSV format expected by this repository by extracting:
- qid
- map
- recip_rank
- ndcg

---

## Minimal Checklist

Before running the main pipeline, confirm these exist:

- dataset/v1/queries.train.small.tsv
- dataset/bm25_eval/v1/train_eval.tsv
- dataset/trec/2019/queries.tsv (if evaluating on 2019)
- dataset/bm25_eval/v1/2019_eval.tsv (if evaluating on 2019)