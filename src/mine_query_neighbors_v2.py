import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse

def load_queries(file_path):
    queries = []
    query_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            query_ids.append(qid)
            queries.append(qtext)
    return query_ids, queries

def save_similarities(out_path, output_file_json, search_query_ids, scores, indices, main_query_ids):
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, (sim_scores, neighbors) in enumerate(zip(scores, indices)):
            search_qid = search_query_ids[i]
            for j, neighbor_idx in enumerate(neighbors):
                sim_qid = main_query_ids[neighbor_idx]
                score = sim_scores[j]
                f.write(f"{search_qid}\t{sim_qid}\t{score:.4f}\n")


    NNQ_dict = {}
    import json

    with open(out_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            query_id, query_id_sim, similarity = parts
            if query_id not in NNQ_dict:
                NNQ_dict[query_id] = {}
                i = 1
            NNQ_dict[query_id][str(i)] = [query_id_sim, float(similarity)]
            i += 1

    with open(output_file_json, 'w') as f:
        json.dump(NNQ_dict, f)
    
def compute_query_similarities(query_file_main, query_file_search, output_file, output_file_json, model_name, top_k=100):
    # Load queries
    main_query_ids, main_queries = load_queries(query_file_main)
    search_query_ids, search_queries = load_queries(query_file_search)
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Encode main queries (index side)
    print("Encoding main queries...")
    main_embeddings = model.encode(
        main_queries, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )
    
    # Encode search queries (search side)
    print("Encoding search queries...")
    search_embeddings = model.encode(
        search_queries, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )

    # Create FAISS index
    d = main_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity (with normalized vectors)
    index.add(main_embeddings)

    # Search
    print("Searching nearest neighbors...")
    scores, indices = index.search(search_embeddings, top_k)

    # Save
    print("Saving results...")
    save_similarities(output_file, output_file_json, search_query_ids, scores, indices, main_query_ids)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute query similarities with FAISS + SentenceTransformer")
    parser.add_argument("--query_file_main", type=str, required=True, help="TSV file with main queries (index side)")
    parser.add_argument("--query_file_search", type=str, required=True, help="TSV file with search queries")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save similarities")
    parser.add_argument("--output_file_json", type=str, required=True, help="Output file to save similarities")

    parser.add_argument("--model_name", type=str, default="sentence-transformers/msmarco-distilbert-base-tas-b",
                        help="SentenceTransformer model name")
    parser.add_argument("--top_k", type=int, default=100, help="Number of nearest neighbors to retrieve")

    args = parser.parse_args()

    compute_query_similarities(
        args.query_file_main,
        args.query_file_search,
        args.output_file,
        args.output_file_json,
        args.model_name,
        top_k=args.top_k
    )