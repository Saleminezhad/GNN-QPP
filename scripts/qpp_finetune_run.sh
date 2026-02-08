#!/bin/bash
#SBATCH --job-name=qpp_finetune
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=logs/qpp_finetune_%j.out
#SBATCH --error=logs/qpp_finetune_%j.err

echo "Starting finetune (performance alignment) pipeline at $(date)"

# ============================================================
# Environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qpp

# ============================================================
# Configuration
# ============================================================
MODEL_NAME=distilbert
BASE_ENCODER=sentence-transformers/msmarco-distilbert-base-v4
METRIC_KEY=ndcg

# Triplet / contrastive settings
MAX_TRIPLETS_PER_QUERY=10
LR=2e-5
TAU_POS=0.05
TAU_NEG=0.15
TOPK_NEIGHBORS=50
EPOCHS=2
BATCH_SIZE=64

TOPK_NNQ=100

# Paths (repo-relative)
DATA_DIR=dataset
BASELINE_DIR=artifacts/BM25_V1/${MODEL_NAME}              # produced by Phase 1 baseline
FT_DIR=artifacts/finetune/${MODEL_NAME}                  # finetuned model + derived artifacts

FT_MODEL_DIR=${FT_DIR}/checkpoints/qpp_${MODEL_NAME}_ft
NNQ_DIR=${FT_DIR}/nnq
OUT_MOTHER_DIR=${FT_DIR}/mother

mkdir -p logs ${FT_MODEL_DIR} ${NNQ_DIR} ${OUT_MOTHER_DIR}

# ============================================================
# Safety checks
# ============================================================
if [ ! -f "${BASELINE_DIR}/MotherDataset_BM25_V1.json" ]; then
  echo "[ERROR] Missing ${BASELINE_DIR}/MotherDataset_BM25_V1.json"
  echo "Run Phase 1 baseline first to create the MotherDataset."
  exit 1
fi

# ============================================================
# Step 1: Fine-tune bi-encoder (performance alignment)
# ============================================================
echo "Step 1: Fine-tuning bi-encoder..."

python src/finetune_biencoder_qpp.py \
  --train_json "${BASELINE_DIR}/MotherDataset_BM25_V1.json" \
  --metric_key "${METRIC_KEY}" \
  --base_model "${BASE_ENCODER}" \
  --output_dir "${FT_MODEL_DIR}" \
  --max_triplets_per_query "${MAX_TRIPLETS_PER_QUERY}" \
  --lr "${LR}" \
  --tau_pos "${TAU_POS}" \
  --tau_neg "${TAU_NEG}" \
  --top_k_neighbors "${TOPK_NEIGHBORS}" \
  --num_epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

# ============================================================
# Step 2: Re-mine NNQ using the fine-tuned encoder
# ============================================================
echo "Step 2: Mining NNQ (train) with fine-tuned encoder..."

python src/mine_query_neighbors.py \
  --query_file_main "${DATA_DIR}/v1/queries.train.small.tsv" \
  --query_file_search "${DATA_DIR}/v1/queries.train.small.tsv" \
  --output_json "${NNQ_DIR}/NNQ_${MODEL_NAME}_train_V1_ft.json" \
  --encoder "${FT_MODEL_DIR}" \
  --top_k "${TOPK_NNQ}"

echo "Step 2: Mining NNQ (TREC 2019) with fine-tuned encoder..."

python src/mine_query_neighbors.py \
  --query_file_main "${DATA_DIR}/v1/queries.train.small.tsv" \
  --query_file_search "${DATA_DIR}/trec/2019/queries.tsv" \
  --output_json "${NNQ_DIR}/NNQ_${MODEL_NAME}_2019_ft.json" \
  --encoder "${FT_MODEL_DIR}" \
  --top_k "${TOPK_NNQ}"

# ============================================================
# Step 3: Convert baseline MotherDataset -> fine-tuned MotherDataset
# (swap NNQ to the fine-tuned neighbors)
# ============================================================
echo "Step 3: Building fine-tuned MotherDataset (train)..."

python src/convert_mother_dataset_with_nnq.py \
  --nnq_path "${NNQ_DIR}/NNQ_${MODEL_NAME}_train_V1_ft.json" \
  --base_mother_path "${BASELINE_DIR}/MotherDataset_BM25_V1.json" \
  --output_path "${OUT_MOTHER_DIR}/MotherDataset_BM25_V1_finetuned.json"

echo "Step 3: Building fine-tuned MotherDataset (2019)..."

python src/convert_mother_dataset_with_nnq.py \
  --nnq_path "${NNQ_DIR}/NNQ_${MODEL_NAME}_2019_ft.json" \
  --base_mother_path "${BASELINE_DIR}/MotherDataset_BM25_V1_2019.json" \
  --output_path "${OUT_MOTHER_DIR}/MotherDataset_BM25_V1_2019_finetuned.json"

echo "Done at $(date)"