#!/bin/bash

[ -f ../.env ] && . ../.env
set +a

export PYTHONHTTPSVERIFY=0

MODEL_PATH=""

DATASET=""

OUTPUT_PATH=""
mkdir -p $OUTPUT_PATH

echo "Running DeepResearch inference on BrowseComp dataset..."

python3 react_agent.py \
  --model_path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_PATH" \
  --enable_web_search "true" \
  --enable_retriever "false" \
  --enable_calculator "false" \
  --enable_file_parser "false"

echo "Inference started! Check results in $OUTPUT_PATH"