#!/bin/bash
# =============================================================================
# Step 0: Server Setup & Docker Environment
# Run this script ON THE SERVER (ssh <user>@<server>)
# =============================================================================
set -euo pipefail

WORK_DIR="${HOME}/parakeet-basque"
NEMO_IMAGE="nvcr.io/nvidia/nemo:26.02"
CONTAINER_NAME="parakeet-basque"

echo "=== Step 0.1: Check prerequisites ==="
echo "--- Disk space ---"
df -h /
echo ""
echo "--- GPU ---"
nvidia-smi
echo ""
echo "--- Memory ---"
free -h
echo ""

echo "=== Step 0.2: Create working directories ==="
mkdir -p "${WORK_DIR}"/{data,models,scripts,results,logs}
echo "Created directories under ${WORK_DIR}"

echo "=== Step 0.3: Check/Pull NeMo Docker image ==="
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "${NEMO_IMAGE}"; then
    echo "Image ${NEMO_IMAGE} already exists"
else
    echo "Pulling ${NEMO_IMAGE} (this may take 10-30 minutes)..."
    docker pull "${NEMO_IMAGE}"
fi

echo "=== Step 0.4: Remove old NeMo image to save space (optional) ==="
# Uncomment the line below if you want to remove the old 24.07 image
# docker rmi nvcr.io/nvidia/nemo:24.07

echo "=== Step 0.5: Start Docker container ==="
# Stop existing container if running
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Copy .env file to working directory if it exists
if [ -f "${WORK_DIR}/.env" ]; then
    ENV_FILE_FLAG="--env-file ${WORK_DIR}/.env"
    echo "Loading env vars from ${WORK_DIR}/.env"
else
    ENV_FILE_FLAG=""
    echo "WARNING: No .env file found at ${WORK_DIR}/.env"
    echo "  Copy .env.example to .env and fill in WANDB_API_KEY and HF_TOKEN"
fi

docker run -d \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    --name "${CONTAINER_NAME}" \
    ${ENV_FILE_FLAG} \
    -v "${WORK_DIR}:/workspace/parakeet-basque" \
    "${NEMO_IMAGE}" \
    sleep infinity

echo "Container '${CONTAINER_NAME}' started."
echo ""

echo "=== Step 0.6: Verify GPU inside container ==="
docker exec "${CONTAINER_NAME}" nvidia-smi

echo "=== Step 0.7: Check NeMo version ==="
docker exec "${CONTAINER_NAME}" python -c "import nemo; print('NeMo version:', nemo.__version__)"

echo "=== Step 0.8: Install additional packages inside container ==="
docker exec "${CONTAINER_NAME}" pip install -q wandb datasets soundfile librosa

echo ""
echo "============================================="
echo "  Setup complete!"
echo "  Container: ${CONTAINER_NAME}"
echo "  Working dir: ${WORK_DIR} (mounted at /workspace/parakeet-basque)"
echo ""
echo "  Next: Run step 01_download_dataset.sh"
echo "============================================="
