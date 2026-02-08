#!/usr/bin/env python
"""
Fast Query–Query Graph Builder (PyTorch Geometric)
--------------------------------------------------

  x: [num_nodes, emb_dim + 2]
     - Node 0 (query):   [embedding, 0.0, 0.0]
     - Neighbors:        [embedding, ndcg, map]

  y: [2] = [ndcg_query, map_query]

  edge_index: [2, num_edges]  (bidirectional star: query ↔ neighbors)
  edge_weight: [num_edges]    (similarity scores, optionally normalized)

  metadata: { "qid": <str>, "neighbor_ids": [list of neighbor qids] }
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------- Helpers ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mother_path", required=True, help="MotherDataset JSON for current set (train/dev/test)")
    p.add_argument("--neighbor_mother_path", required=True, help="Full MotherDataset JSON containing neighbor queries (usually the training dataset)")
    p.add_argument("--out_dir", required=True, help="Output folder to store graph_<qid>.pt files")
    p.add_argument("--model_name", default="sentence-transformers/msmarco-distilbert-base-v4")
    p.add_argument("--topk_qq", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--normalize_edges", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8, help="Parallel CPU workers for graph saving")
    return p.parse_args()


def load_mother_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    logging.info("Loaded %d queries from %s", len(data), path)
    return data

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def resolve_neighbors(entry, topk):
    """
    Convert entry['query_run'] from:
        {"0": [id, sim], "1": [id, sim], ...}
    to a sorted list:
        [[id, sim], [id, sim], ...]
    and return the top-k neighbors.
    """
    run = entry.get("query_run", {})
    # Convert to list sorted by numeric key
    neighbors = [run[k] for k in sorted(run.keys(), key=lambda x: int(x))]
    return neighbors[:topk]


# ---------------------- Graph Builder ----------------------

def build_graph(qid, entry, mother, neighbor_mother, query_emb_cache, args):
    """
    Builds a single PyG graph for one query.

    Node features:
      - Query node:   [embedding, 0.0, 0.0]
      - Neighbor q's: [embedding, ndcg_neighbor, map_neighbor]

    Label:
      - y = [ndcg_query, map_query]
    """
    query_text = entry.get("query")
    eval_block = entry.get("eval", {})

    if not query_text:
        return None

    # ----- 1) Query-level metrics -----
    
    if "ndcg" not in eval_block:
        raise KeyError(f"Missing 'ndcg' for query {qid} in {args.mother_path}")
    ndcg_q = float(eval_block["ndcg"])

    if "map" not in eval_block:
        raise KeyError(f"Missing 'map' for query {qid} in {args.mother_path}")
    map_q = float(eval_block["map"])
    
    label_vec = np.array([ndcg_q, map_q], dtype=np.float32)  # [2]

    neighbors = resolve_neighbors(entry, args.topk_qq)


    # ----- 2) Query embedding -----
    query_emb = query_emb_cache.get(qid)
    if query_emb is None:
        return None

    # Query node feature: [embedding, 0.0, 0.0]  (no leakage of its own metrics)
    query_features = np.concatenate([query_emb, [0.0, 0.0]])  # [emb_dim + 2]

    node_features = [query_features]
    neighbor_ids = []
    edge_weight = []

    # ----- 3) Neighbor nodes -----
    for item in neighbors:
        nid, score = str(item[0]), float(item[1])

        # neighbor_emb = query_emb_cache[nid]
        neighbor_emb = query_emb_cache.get(nid)
        if neighbor_emb is None:
            # optional: log once in debug mode
            logging.debug(f"Skipping neighbor {nid} for query {qid}: no embedding in cache")
            continue
        
        # Get neighbor's eval block from either dataset
        n_eval = neighbor_mother[nid].get("eval", {})
        n_ndcg = float(n_eval["ndcg"])
        n_map = float(n_eval["map"])

        # Neighbor feature: [embedding, ndcg, map]
        neighbor_features = np.concatenate([neighbor_emb, [n_ndcg, n_map]])

        node_features.append(neighbor_features)
        neighbor_ids.append(nid)
        edge_weight.append(score)  # one weight per neighbor (forward edge)

    if len(neighbor_ids) == 0:
        return None

    # ----- 4) Build edge_index as bidirectional star -----
    edge_index = []
    # Nodes: 0 = query, 1..len(node_features)-1 = neighbors
    for idx in range(1, len(node_features)):
        edge_index.append([0, idx])   # query → neighbor
        edge_index.append([idx, 0])   # neighbor → query

    edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))

    # Duplicate weights for reverse edges
    edge_weight = np.array(edge_weight, dtype=np.float32)
    edge_weight = np.concatenate([edge_weight, edge_weight], axis=0)  # [2 * num_neighbors]

    # ----- 5) Create PyG Data object -----
    x = torch.tensor(np.array(node_features), dtype=torch.float32)   # [num_nodes, emb_dim+2]
    ei = torch.tensor(edge_index, dtype=torch.long)                  # [2, num_edges]
    ew = torch.tensor(edge_weight, dtype=torch.float32)              # [num_edges]
    y = torch.tensor(label_vec, dtype=torch.float32)                 # [2]

    data = Data(
        x=x,
        edge_index=ei,
        edge_weight=ew,
        y=y,
    )

    data.metadata = {"qid": qid, "neighbor_ids": neighbor_ids}
    return data


def build_and_save_graph(qid, entry, mother, neighbor_mother, query_emb_cache, args):
    """Build a single PyG graph and persist it immediately to disk."""
    data = build_graph(qid, entry, mother, neighbor_mother, query_emb_cache, args)
    if data is None:
        return 0, 1

    qid = data.metadata["qid"]
    out_path = os.path.join(args.out_dir, f"graph_{qid}.pt")
    torch.save(data, out_path)
    return 1, 0


# ---------------------- Main ----------------------

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # Load datasets
    mother = load_mother_dataset(args.mother_path)
    if os.path.abspath(args.mother_path) == os.path.abspath(args.neighbor_mother_path):
        neighbor_mother = mother
    else:
        neighbor_mother = load_mother_dataset(args.neighbor_mother_path)

    items = list(mother.items())
    if args.limit:
        items = items[:args.limit]

    # ---------------- Encode Queries ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Loading model %s on %s", args.model_name, device)
    model = SentenceTransformer(args.model_name, device=device)
    model.eval()
    torch.set_grad_enabled(False)

    # Collect all queries from BOTH datasets
    all_qids = list({**mother, **neighbor_mother}.keys())
    all_queries = []
    for qid in all_qids:
        qtext = mother.get(qid, {}).get("query") or neighbor_mother.get(qid, {}).get("query", "")
        all_queries.append(qtext)

    logging.info("Encoding %d unique queries (mother + neighbor_mother)...", len(all_qids))
    query_emb_matrix = model.encode(
        all_queries,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_emb_cache = {qid: emb for qid, emb in zip(all_qids, query_emb_matrix)}

    # ---------------- Build Graphs ----------------
    logging.info("Building PyG graphs (streaming saves)...")

    saved, skipped = 0, 0
    if args.num_workers <= 1:
        for qid, entry in tqdm(items, desc="Graphs", ncols=100):
            s, k = build_and_save_graph(qid, entry, mother, neighbor_mother, query_emb_cache, args)
            saved += s
            skipped += k
    else:
        results = Parallel(n_jobs=args.num_workers, prefer="threads")(
            delayed(build_and_save_graph)(qid, entry, mother, neighbor_mother, query_emb_cache, args)
            for qid, entry in tqdm(items, desc="Graphs", ncols=100)
        )
        saved = sum(s for s, _ in results)
        skipped = sum(k for _, k in results)

    logging.info("✅ Saved %d graphs | Skipped %d (missing neighbors or labels)", saved, skipped)
    logging.info("Graphs saved to %s", args.out_dir)


if __name__ == "__main__":
    main()