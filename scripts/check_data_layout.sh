#!/bin/bash

echo "Checking dataset layout for KGQPP..."
echo

ERROR=0

check_file () {
  if [ ! -f "$1" ]; then
    echo "[MISSING] $1"
    ERROR=1
  else
    echo "[OK]      $1"
  fi
}

# --------------------------------------------------
# Required query files
# --------------------------------------------------

check_file dataset/v1/queries.train.small.tsv
check_file dataset/trec/2019/queries.tsv

# --------------------------------------------------
# Required BM25 evaluation files
# --------------------------------------------------

check_file dataset/bm25_eval/v1/train_eval.tsv
check_file dataset/bm25_eval/v1/2019_eval.tsv

echo
if [ $ERROR -eq 1 ]; then
  echo "Dataset check FAILED."
  echo "See dataset/README.md for instructions."
  exit 1
else
  echo "Dataset check PASSED."
fi

