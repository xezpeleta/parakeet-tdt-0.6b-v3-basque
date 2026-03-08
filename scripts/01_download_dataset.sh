#!/bin/bash
# =============================================================================
# Step 1: Download dataset and prepare NeMo manifests
# Run this ON THE SERVER after 00_setup.sh
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"

echo "=== Step 1: Download & prepare dataset ==="
echo "This will download ~77GB of audio data and create NeMo manifests."
echo "Estimated time: 30-90 minutes depending on connection speed."
echo ""

# The Python script is already at scripts/prepare_dataset.py
# (mounted inside the container at /workspace/parakeet-basque/scripts/)

echo "Running dataset preparation inside container..."
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/prepare_dataset.py

echo ""
echo "============================================="
echo "  Dataset download & preparation complete!"
echo "  Next: Run 02_baseline_eval.sh"
echo "============================================="
