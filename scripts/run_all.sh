#!/bin/bash
# =============================================================================
# Master script: Run all steps sequentially
# Run this ON THE SERVER or use individual step scripts
#
# Usage:
#   ./run_all.sh           # Run all steps
#   ./run_all.sh 3         # Start from step 3 (fine-tuning)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_STEP="${1:-0}"

steps=(
    "00_setup.sh"
    "01_download_dataset.sh"
    "01b_create_tarred_dataset.sh"
    "02_baseline_eval.sh"
    "03_finetune.sh"
    "04_evaluate.sh"
    "05_export.sh"
)

step_names=(
    "Setup Docker environment"
    "Download & prepare dataset"
    "Create tarred dataset"
    "Baseline evaluation"
    "Fine-tuning"
    "Evaluate fine-tuned model"
    "Export model"
)

echo "============================================="
echo "  Parakeet TDT 0.6B v3 — Basque Fine-tuning"
echo "============================================="
echo ""

for i in "${!steps[@]}"; do
    if [ "$i" -lt "$START_STEP" ]; then
        echo "[SKIP] Step $i: ${step_names[$i]}"
        continue
    fi

    echo ""
    echo "=============================="
    echo "[STEP $i] ${step_names[$i]}"
    echo "=============================="
    echo ""

    bash "${SCRIPT_DIR}/${steps[$i]}"

    echo ""
    echo "[STEP $i] ${step_names[$i]} — DONE"
done

echo ""
echo "============================================="
echo "  All steps complete!"
echo "============================================="
