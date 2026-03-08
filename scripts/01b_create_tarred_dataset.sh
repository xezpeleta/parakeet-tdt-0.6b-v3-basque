#!/bin/bash
# =============================================================================
# Step 1b: Convert training data to tarred format for efficient training
# Run this ON THE SERVER after 01_download_dataset.sh
# =============================================================================
set -euo pipefail

CONTAINER_NAME="parakeet-basque"
DATA_DIR="/workspace/parakeet-basque/data"

echo "=== Step 1b: Convert training data to tarred dataset ==="
echo "This improves training I/O performance significantly."
echo "Estimated time: 15-30 minutes."
echo ""

docker exec "${CONTAINER_NAME}" bash -c "
    NEMO_ROOT=\$(python -c 'import nemo; import os; print(os.path.dirname(nemo.__file__) + \"/../\")' 2>/dev/null)
    
    # Find the convert script
    CONVERT_SCRIPT=\"\"
    for path in \
        \"\${NEMO_ROOT}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py\" \
        /opt/NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
        /workspace/NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py; do
        if [ -f \"\$path\" ]; then
            CONVERT_SCRIPT=\"\$path\"
            break
        fi
    done
    
    if [ -z \"\$CONVERT_SCRIPT\" ]; then
        echo 'ERROR: convert_to_tarred_audio_dataset.py not found!'
        echo 'Searching...'
        find / -name 'convert_to_tarred_audio_dataset.py' 2>/dev/null | head -5
        exit 1
    fi
    
    echo \"Using script: \$CONVERT_SCRIPT\"
    
    # Only convert train split (dev/test remain as regular manifests)
    python \"\$CONVERT_SCRIPT\" \
        --manifest_path=${DATA_DIR}/train_manifest.json \
        --target_dir=${DATA_DIR}/train_tarred \
        --num_shards=512 \
        --max_duration=20.0 \
        --min_duration=0.5 \
        --shuffle \
        --shuffle_seed=42 \
        --sort_in_shards \
        --workers=-1
    
    echo ''
    echo 'Tarred dataset created at ${DATA_DIR}/train_tarred/'
    ls -lh ${DATA_DIR}/train_tarred/ | head -20
    echo '...'
    echo \"Total tar files: \$(ls ${DATA_DIR}/train_tarred/*.tar 2>/dev/null | wc -l)\"
"

echo ""
echo "============================================="
echo "  Tarred dataset creation complete!"
echo "  Next: Run 02_baseline_eval.sh"
echo "============================================="
