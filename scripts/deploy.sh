#!/bin/bash
# =============================================================================
# Deploy scripts to server and optionally start the pipeline
#
# Usage (from your local machine):
#   ./deploy.sh              # Just copy files
#   ./deploy.sh --run        # Copy files and start run_all.sh in tmux
#   ./deploy.sh --run 3      # Copy files and start from step 3
# =============================================================================
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Read server config (not tracked in git — copy config/server.conf.example to config/server.conf)
SERVER_CONF="${LOCAL_DIR}/config/server.conf"
if [ ! -f "${SERVER_CONF}" ]; then
    echo "ERROR: config/server.conf not found."
    echo "  Copy config/server.conf.example to config/server.conf and fill in your server details."
    exit 1
fi
HOST=$(grep "^host" "${SERVER_CONF}" | awk -F'=' '{print $2}' | tr -d ' ')
USERNAME=$(grep "^username" "${SERVER_CONF}" | awk -F'=' '{print $2}' | tr -d ' ')
SERVER="${USERNAME}@${HOST}"
REMOTE_DIR="/home/${USERNAME}/parakeet-basque"

echo "=== Deploying to ${SERVER}:${REMOTE_DIR} ==="

# Create remote directories
ssh "${SERVER}" "mkdir -p ${REMOTE_DIR}/{scripts,config}"

# Copy scripts
echo "Copying scripts..."
scp "${LOCAL_DIR}"/scripts/*.sh "${SERVER}:${REMOTE_DIR}/scripts/"

# Copy .env.example (user must create .env manually)
scp "${LOCAL_DIR}/.env.example" "${SERVER}:${REMOTE_DIR}/"

# Make scripts executable
ssh "${SERVER}" "chmod +x ${REMOTE_DIR}/scripts/*.sh"

echo "Files deployed to ${SERVER}:${REMOTE_DIR}"
echo ""

# Check if .env exists on server
if ssh "${SERVER}" "test -f ${REMOTE_DIR}/.env"; then
    echo ".env file found on server."
else
    echo "WARNING: No .env file on server!"
    echo "  1. SSH into the server: ssh ${SERVER}"
    echo "  2. Copy and edit: cp ${REMOTE_DIR}/.env.example ${REMOTE_DIR}/.env"
    echo "  3. Fill in WANDB_API_KEY and HF_TOKEN"
    echo ""
fi

# Optionally start the pipeline
if [ "${1:-}" = "--run" ]; then
    START_STEP="${2:-0}"
    echo "Starting pipeline from step ${START_STEP} in tmux session 'parakeet'..."
    ssh "${SERVER}" "tmux kill-session -t parakeet 2>/dev/null || true; tmux new-session -d -s parakeet 'cd ${REMOTE_DIR} && bash scripts/run_all.sh ${START_STEP} 2>&1 | tee logs/run_\$(date +%Y%m%d_%H%M%S).log'"
    echo ""
    echo "Pipeline started in tmux session 'parakeet'."
    echo "To monitor:"
    echo "  ssh ${SERVER} -t 'tmux attach -t parakeet'"
    echo ""
    echo "To check logs:"
    echo "  ssh ${SERVER} 'tail -50 ${REMOTE_DIR}/logs/*.log'"
else
    echo "To start the pipeline:"
    echo "  ssh ${SERVER}"
    echo "  cd ${REMOTE_DIR}"
    echo "  tmux new -s parakeet"
    echo "  bash scripts/run_all.sh"
fi
