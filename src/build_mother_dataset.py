# read .tsv file
import argparse
import csv
import json
import logging
import os
from collections import defaultdict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Build MotherDataset with minimal edits")
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--nnq_path", required=True)  # sim.search.py output
    parser.add_argument("--eval_path", required=True)  # faiss utils eval tsv
    parser.add_argument("--main_path", required=True)  # saved folder
    parser.add_argument("--graph_dataset_path", required=True)
    args = parser.parse_args()

    query_path = args.query_path
    NNQ = args.nnq_path
    eval_path = args.eval_path
    main_path = args.main_path
    graph_dataset_path = args.graph_dataset_path

    class QueryIndex:
        def __init__(self, file_path):
            self.file_path = file_path
            self.query_data = {}
            self.load_data()
            
        def tsvToDict(self):
            with open(self.file_path, 'rt') as file:
                for line in file:
                    parts = line.rstrip('\n').split('\t', 1)
                    if len(parts) == 2:
                        query_id, query_text = parts
                        self.query_data[query_id] = query_text
                        
        def load_data(self):
            if os.path.exists(self.file_path):
                self.tsvToDict()
            else:
                logging.error(f"File {self.file_path} does not exist.")

        def get_qtext(self, query_id):
            return self.query_data.get(query_id, None)
        
    query_data = QueryIndex(query_path)

    with open(NNQ, 'r') as f:
        NNQ_dict = json.load(f)

    def convert_eval_tsv_to_dict(input_path):
        eval_data = {}

        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                qid = row['']  # keep as-is
                eval_data[qid] = {
                    'map': row['map'],
                    'rr': row['recip_rank'],
                    'ndcg': row['ndcg']
                }

        return eval_data

    if not os.path.exists(main_path):
        os.makedirs(main_path, exist_ok=True)
    eval_data = convert_eval_tsv_to_dict(eval_path)

    MotherDataset = {}
    i = 0
    skipped_missing = 0
    for query_id in eval_data.keys():  # minimal change to iterate keys
        i += 1
        if i % 100000 == 0:
            logging.info(f"Processed {i} queries")

        query_text = query_data.query_data.get(query_id)
        query_neighbors = NNQ_dict.get(query_id, [])
            
        MotherDataset[query_id] = {}
        
        MotherDataset[query_id]["query"] = query_text
        MotherDataset[query_id]["eval"] = eval_data[query_id]
        MotherDataset[query_id]['query_run'] = query_neighbors

    out_file = os.path.join(main_path, graph_dataset_path)
    with open(out_file, 'w') as f:
        json.dump(MotherDataset, f)
    logging.info(
        "Mother dataset saved to %s | total=%d | kept=%d | skipped=%d",
        out_file,
        len(eval_data),
        len(MotherDataset),
        skipped_missing,
    )

if __name__ == "__main__":
    main()
