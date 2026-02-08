#!/usr/bin/env python
"""
Fine-tune SentenceTransformer (e.g., msmarco-distilbert-base-v4)
as a QPP-aware encoder using your JSON format and TripletLoss.

JSON format (train):
{
  "19335": {
    "query": "anthropological definition of environment",
    "eval": {"map": "0.2847", "rr": "1.0", "ndcg": "0.7198"},
    "query_run": {
      "1": ["19337", 0.8863],
      "2": ["113558", 0.8851],
      "3": ["19338", 0.8844],
      ...
    }
  },
  ...
}
"""
import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_json_data(
    json_path: str,
    metric_key: str,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, List]]]:
    """
    Returns:
      - qid2text: qid -> query string
      - qid2score: qid -> float score (metric_key)
      - qid2neighbors: qid -> raw query_run dict
    """
    logger.info("Loading JSON from %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, list):
        data = {str(x["qid"]): x for x in raw}
    else:
        raise ValueError("Unsupported JSON structure; expected dict or list.")

    qid2text = {}
    qid2score = {}
    qid2neighbors = {}

    missing_metric = 0

    for qid, rec in data.items():
        qid = str(qid)
        query = rec.get("query", "").strip()
        eval_dict = rec.get("eval", {})
        query_run = rec.get("query_run", {})

        if not query:
            continue

        if metric_key not in eval_dict:
            missing_metric += 1
            continue

        try:
            score = float(eval_dict[metric_key])
        except (ValueError, TypeError):
            missing_metric += 1
            continue

        qid2text[qid] = query
        qid2score[qid] = score
        qid2neighbors[qid] = query_run

    if missing_metric > 0:
        logger.warning(
            "Missing/invalid '%s' for %d queries; those were skipped.",
            metric_key, missing_metric
        )

    logger.info(
        "Loaded %d queries with text + '%s' + neighbors.",
        len(qid2text), metric_key
    )
    return qid2text, qid2score, qid2neighbors


def build_triplets(
    qid2text: Dict[str, str],
    qid2score: Dict[str, float],
    qid2neighbors: Dict[str, Dict[str, List]],
    tau_pos: float,
    tau_neg: float,
    max_triplets_per_query: int,
    top_k_neighbors: int,
    seed: int = 42,
) -> List[InputExample]:
    """
    Build InputExample(texts=[anchor, pos, neg]) triplets.
    """
    random.seed(seed)
    examples: List[InputExample] = []
    skipped_no_valid = 0

    for qid, query_run in qid2neighbors.items():
        if qid not in qid2score:
            continue

        anchor_score = qid2score[qid]
        anchor_text = qid2text[qid]

        # query_run: rank -> [neighbor_qid, sim]
        # sort by rank and truncate to top_k_neighbors
        items = []
        for rk, pair in query_run.items():
            try:
                rank_int = int(rk)
            except ValueError:
                continue
            if not isinstance(pair, list) or len(pair) < 1:
                continue
            neighbor_qid = str(pair[0])
            items.append((rank_int, neighbor_qid))

        items.sort(key=lambda x: x[0])
        items = items[:top_k_neighbors]

        pos_cands = []
        neg_cands = []

        for _, n_qid in items:
            if n_qid not in qid2score:
                continue
            diff = abs(anchor_score - qid2score[n_qid])
            if diff <= tau_pos:
                pos_cands.append(n_qid)
            elif diff >= tau_neg:
                neg_cands.append(n_qid)

        if not pos_cands or not neg_cands:
            skipped_no_valid += 1
            continue

        num_triplets = min(
            max_triplets_per_query,
            len(pos_cands) * len(neg_cands)
        )

        for _ in range(num_triplets):
            p_qid = random.choice(pos_cands)
            n_qid = random.choice(neg_cands)

            pos_text = qid2text[p_qid]
            neg_text = qid2text[n_qid]

            examples.append(
                InputExample(texts=[anchor_text, pos_text, neg_text])
            )

    if skipped_no_valid > 0:
        logger.warning(
            "Skipped %d anchors without both positive and negative neighbors.",
            skipped_no_valid
        )

    logger.info("Constructed %d triplets.", len(examples))
    if len(examples) == 0:
        raise RuntimeError("No triplets built; check thresholds and data.")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer as QPP-aware encoder using JSON + TripletLoss.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to train JSON (with query, eval, query_run).")
    parser.add_argument("--metric_key", type=str, default="ndcg", help="Eval metric key to use as score (e.g., ndcg, map).")
    parser.add_argument("--base_model", type=str, default="sentence-transformers/msmarco-distilbert-base-v4")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned SentenceTransformer.")
    parser.add_argument("--tau_pos", type=float, default=0.05, help="Max score diff for positive neighbors.")
    parser.add_argument("--tau_neg", type=float, default=0.15, help="Min score diff for negative neighbors.")
    parser.add_argument("--max_triplets_per_query", type=int, default=4)
    parser.add_argument("--top_k_neighbors", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.info("Args: %s", args)

    # 1) Load data
    qid2text, qid2score, qid2neighbors = load_json_data(
        args.train_json,
        metric_key=args.metric_key,
    )

    train_examples = build_triplets(
        qid2text=qid2text,
        qid2score=qid2score,
        qid2neighbors=qid2neighbors,
        tau_pos=args.tau_pos,
        tau_neg=args.tau_neg,
        max_triplets_per_query=args.max_triplets_per_query,
        top_k_neighbors=args.top_k_neighbors,
        seed=args.seed,
    )

    # 2) Build DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # 3) Load base SentenceTransformer
    model = SentenceTransformer(args.base_model)

    # 4) Triplet loss on sentence embeddings
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.2,
    )

    # 5) Training
    num_train_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(args.warmup_ratio * num_train_steps)

    logging.info("Num train steps: %d | warmup_steps: %d", num_train_steps, warmup_steps)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )

    # 6) Save fine-tuned SentenceTransformer model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir)
    logging.info("Saved fine-tuned model to %s", args.output_dir)


if __name__ == "__main__":
    main()