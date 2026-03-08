#!/bin/bash
# =============================================================================
# Step 5: Export model and optionally push to HuggingFace
# Run this ON THE SERVER after 04_evaluate.sh
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"

echo "=== Step 5: Export model ==="
echo ""

# The Python script is at scripts/export_model.py (mounted in container)

docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/export_model.py

echo ""
echo "============================================="
echo "  Export complete!"  
echo "  Model: ${HOME}/parakeet-basque/models/parakeet-tdt-0.6b-v3-basque.nemo"
echo "  Card:  ${HOME}/parakeet-basque/models/README.md"
echo "============================================="
