#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="parakeet-basque"

echo "============================================================"
echo "Step 06: Export ONNX"
echo "============================================================"

echo "Running ONNX export inside container: ${CONTAINER_NAME}"
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/export_onnx.py

echo

echo "Done. Expected output:"
echo "  ${HOME}/parakeet-basque/models/parakeet-tdt-0.6b-v3-basque.onnx"
