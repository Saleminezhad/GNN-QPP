# GNN-QPP

## Dataset
1. Download datasets

MS MARCO Passage V2:

`wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar`

For more information get to this website:

'https://microsoft.github.io/msmarco/TREC-Deep-Learning.html'

MS MARCO Passage V1:

`wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz`

we need to generate a graph dataset consisting of all the queries and documents. for the MS MARCO V1 dataset. 
• Documents: 8,841,823 web documents (in the full document collection).
• Queries: 808,731 unique queries.
• Relevance Judgments: 532,761 (labelled query-document pairs).

2.	Graph Structure:

Nodes: 

- Queries
  - different attribution can be assigned to this  like relevance score
  - first assumption is to generate the embedding and assign the embedding as the initial value for the 
- Documents
  - Same embeddings generate by the dense retriever can be used here

Edges:

- Query-Query Similarity: Captures semantic or lexical similarity between queries. 
  - similarity can be measured by different cross-encoder models that is used for semantic search.
  - models: sentence-transformers/msmarco-MiniLM-L6-cos-v5, 
  - we will save the embedding to save the computation cost
- Document-Document Similarity: Helps model document relationships based on content, embeddings, or retrieval scores.
  - same model for the Query-Query can be  used here
- Query-Document Retrieval Relationship: Links queries to their retrieved documents, weighted by retrieval scores or ranking positions.
  - similarity score
  - relevance score (since we are using the dense retrievers we can say that the relevance score is actually the similarity score)


3. Dense Retrievers

- Models for dense retrievers for indexing documents
  - "msmarco-distilbert-base-tas-b"
  - "msmarco-roberta-base-ance-firstp"
  - "msmarco-MiniLM-L6-cos-v5" 
  - "colbert-ir/colbertv2.0" it can be used from its own library not sentence transformer


- Models for dense retrievers for indexing queries
  - "msmarco-MiniLM-L6-cos-v5"
  - "sentence-transformers/all-MiniLM-L6-v2"`


## Index, Retrieve and Evaluation

These function have been implemented for using faiss library for dense retriever. parallel computation is used to reduce the run time.

you can use this header if you are using the slurm:

'''#!/bin/bash
#SBATCH --job-name=multi_search             # Job name
#SBATCH --partition=main              # Partition name
#SBATCH --output=output_bash/faiss_index_%j.txt
#SBATCH --job-name=Parallel_FAISS_Retrieval
#SBATCH --gres=gpu:rtx6000ada:1     # Request 1 GPU
#SBATCH --ntasks=1                   # Only 1 task (we handle parallelism in Python)
#SBATCH --cpus-per-task=80             # Use 8 CPUs (adjust based on availability)
#SBATCH --mem=120G                     # Increase memory to prevent OOM errors
#SBATCH --error=output_bash/err_faiss_index_%j.txt         # Separate file for standard error'''

### Index


'echo "Starting faiss_utils.py with multiprocessing..."
source /mnt/data/abbas/miniconda3/etc/profile.d/conda.sh
conda activate gnn

export model_name="msmarco-distilbert-base-tas-b" #"all-MiniLM-L6-v2" # "msmarco-MiniLM-L6-cos-v5" , "msmarco-roberta-base-ance-firstp"
export collection="v1" # v1 or v2

python faiss_utils.py --encode True \
                    --num_threads 2 \
                    --chunk_size_encode 1000000 \
                    --model_name $model_name  \
                    --input_encode_file dataset/$collection/collection.tsv \
                    --index_path faiss/faiss_index/$collection/doc_$model_name/faiss_index
'

### Retrieve

'''
echo "Starting faiss_utils.py with multiprocessing..."
source /mnt/data/abbas/miniconda3/etc/profile.d/conda.sh
conda activate gnn

python faiss_utils.py --retrieve True \
                    --num_threads 2 \
                    --chunk_size_retrieve 10000 \
                    --model_name $model_name  \
                    --top_k 1000 \
                    --queries_file dataset/$collection/queries.train.tsv \
                    --output_run_file faiss/faiss_run/$collection/doc_$model_name/queries.train.tsv/run_file \
                    --index_path faiss/faiss_index/$collection/doc_$model_name/faiss_index
'''

### evaluate

by defult it calculate these metric @ 1000
metrics: {'map', 'ndcg', 'recip_rank'}
you can set the rank_cut to calculate these at different rank cuts like 10.

'''
echo "Starting faiss_utils.py with multiprocessing..."
source /mnt/data/abbas/miniconda3/etc/profile.d/conda.sh
conda activate gnn

python faiss_utils.py --evaluate True \
                    --num_threads 4 \
                    --qrels_file dataset/$collection/qrels.train.tsv \
                    --model_name $model_name \
                    --queries_file dataset/$collection/queries.train.tsv \
                    --output_eval_file faiss/faiss_eval/$collection/doc_$model_name/queries.train.tsv/eval_file \
                    --chunk_size_eval 20000 \
'''