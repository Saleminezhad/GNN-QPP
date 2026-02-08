#!/bin/bash
#SBATCH --job-name=qpp_run
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=logs/qpp_%j.out
#SBATCH --error=logs/qpp_%j.err

echo "Starting QPP pipeline at $(date)"

# ============================================================
# Environment
# ============================================================

# Assumes environment.yml is provided in the repo
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qpp

# ============================================================
# Configuration
# ============================================================

MODEL_NAME=distilbert
BASE_ENCODER=sentence-transformers/msmarco-distilbert-base-v4

TOPK_NNQ=100
TOPK_QQ=25
LABEL_METRIC=ndcg
TRAIN_METRIC=ndcg

DATA_DIR=dataset
OUT_DIR=artifacts/BM25_V1/${MODEL_NAME}

mkdir -p ${OUT_DIR}

# ============================================================
# Step 1: Nearest-neighbor queries (query–query similarity)
# ============================================================

echo "Computing nearest-neighbor queries (train)..."

python src/mine_query_neighbors.py \
  --query_file_main ${DATA_DIR}/v1/queries.train.small.tsv \
  --query_file_search ${DATA_DIR}/v1/queries.train.small.tsv \
  --output_json ${DATA_DIR}/nnq/NNQ_${MODEL_NAME}_train.json \
  --encoder ${BASE_ENCODER} \
  --top_k ${TOPK_NNQ}

echo "Computing nearest-neighbor queries (TREC 2019)..."

python src/mine_query_neighbors.py \
  --query_file_main ${DATA_DIR}/v1/queries.train.small.tsv \
  --query_file_search ${DATA_DIR}/trec/2019/queries.tsv \
  --output_json ${DATA_DIR}/nnq/NNQ_${MODEL_NAME}_2019.json \
  --encoder ${BASE_ENCODER} \
  --top_k ${TOPK_NNQ}

# ============================================================
# Step 2: Build MotherDataset (BM25 supervision)
# ============================================================

echo "Building MotherDataset (train)..."

python src/build_mother_dataset.py \
  --query_path ${DATA_DIR}/v1/queries.train.small.tsv \
  --nnq_path ${DATA_DIR}/nnq/NNQ_${MODEL_NAME}_train.json \
  --eval_path ${DATA_DIR}/bm25_eval/v1/train_eval.tsv \
  --output_path ${OUT_DIR}/MotherDataset_BM25_V1.json

echo "Building MotherDataset (2019)..."

python src/build_mother_dataset.py \
  --query_path ${DATA_DIR}/trec/2019/queries.tsv \
  --nnq_path ${DATA_DIR}/nnq/NNQ_${MODEL_NAME}_2019.json \
  --eval_path ${DATA_DIR}/bm25_eval/v1/2019_eval.tsv \
  --output_path ${OUT_DIR}/MotherDataset_BM25_V1_2019.json

# ============================================================
# Step 3: Build PyG graphs
# ============================================================

echo "Building PyG graphs..."

python src/build_pyg_graphs.py \
  --mother_path ${OUT_DIR}/MotherDataset_BM25_V1.json \
  --out_dir ${OUT_DIR}/pyg_graphs \
  --topk_qq ${TOPK_QQ} \
  --label_metric ${LABEL_METRIC}

# ============================================================
# Step 4: Train GNN regressor
# ============================================================

echo "Training GNN..."

python src/train_gnn_qpp.py \
  --graphs_dir ${OUT_DIR}/pyg_graphs \
  --test_graphs ${OUT_DIR}/pyg_graphs_2019 \
  --epochs 8 \
  --hidden_dim 128 \
  --gnn gat \
  --gat_heads 4 \
  --feature_mode emb_only \
  --label_metric ${LABEL_METRIC} \
  --train_metric ${TRAIN_METRIC} \
  --batch_size 512 \
  --output_dir results/qpp_${MODEL_NAME}

# ============================================================
# Step 5: Correlation analysis
# ============================================================

python src/evaluate_correlation.py \
  --pred_dir results/qpp_${MODEL_NAME} \
  --collection V1 \
  --save results/correlation_${MODEL_NAME}

echo "Pipeline finished at $(date)"