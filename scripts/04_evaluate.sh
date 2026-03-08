#!/bin/bash
# =============================================================================
# Step 4: Evaluate fine-tuned model on all test splits
# Run this ON THE SERVER after 03_finetune.sh
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"

echo "=== Step 4: Evaluate fine-tuned model ==="
echo ""

# The Python script is at scripts/evaluate_finetuned.py (mounted in container)

docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/evaluate_finetuned.py

echo ""
echo "============================================="
echo "  Evaluation complete!"
echo "  Next: Run 05_export.sh to export the model"
echo "============================================="
