#!/bin/bash
# =============================================================================
# Step 2: Baseline evaluation — measure WER of pretrained model on Basque
# Run this ON THE SERVER after 01_download_dataset.sh
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"
DATA_DIR="/workspace/parakeet-basque/data"
RESULTS_DIR="/workspace/parakeet-basque/results"
MODEL_NAME="nvidia/parakeet-tdt-0.6b-v3"

echo "=== Step 2: Baseline evaluation ==="
echo "Running pretrained ${MODEL_NAME} on Basque test sets."
echo "This establishes the WER before fine-tuning."
echo ""

# The Python script is at scripts/baseline_eval.py (mounted in container)

echo "Running baseline evaluation inside container..."
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/baseline_eval.py

echo ""
echo "============================================="
echo "  Baseline evaluation complete!"
echo "  Check results in ${HOME}/parakeet-basque/results/"
echo "  Next: Run 03_finetune.sh"
echo "============================================="
