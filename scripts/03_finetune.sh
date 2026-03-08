#!/bin/bash
# =============================================================================
# Step 3: Fine-tune parakeet-tdt-0.6b-v3 on Basque data
# Run this ON THE SERVER after 01b_create_tarred_dataset.sh and 02_baseline_eval.sh
#
# IMPORTANT: Make sure .env file is configured with WANDB_API_KEY
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"
DATA_DIR="/workspace/parakeet-basque/data"
RESULTS_DIR="/workspace/parakeet-basque/results"
MODEL_NAME="nvidia/parakeet-tdt-0.6b-v3"

echo "=== Step 3: Fine-tuning ==="
echo "Fine-tuning ${MODEL_NAME} on Basque data."
echo "This will take several hours on L40 GPU."
echo ""

# Check if tarred dataset exists
docker exec "${CONTAINER_NAME}" bash -c '
if [ -d "/workspace/parakeet-basque/data/train_tarred" ] && [ -f "/workspace/parakeet-basque/data/train_tarred/tarred_audio_manifest.json" ]; then
    echo "Using tarred dataset for training."
else
    echo "WARNING: Tarred dataset not found. Using regular manifest (slower)."
    echo "Consider running 01b_create_tarred_dataset.sh first."
fi
'

echo "Starting fine-tuning (this will run for several hours)..."
echo "Use Ctrl+C to stop. Training can be resumed thanks to checkpoint saving."
echo ""

# The Python script is at scripts/finetune.py (mounted in container)
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/finetune.py

echo ""
echo "============================================="
echo "  Fine-tuning complete!"
echo "  Next: Run 04_evaluate.sh"
echo "============================================="
