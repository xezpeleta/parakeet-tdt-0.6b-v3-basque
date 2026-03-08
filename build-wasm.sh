#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Building Rust WASM with Docker..."
echo "    (first run downloads Rust crates; subsequent runs are fast)"

# Build the Docker image (layer-cached after first run)
docker build -f Dockerfile.wasm -t parakeet-wasm-builder .

# Create output directory
mkdir -p docs/pkg

# Run the container: mount source read-only, output directory read-write
# The container already compiled everything during `docker build` and placed
# output in /output. We use docker cp to grab the artifacts.
CONTAINER_ID=$(docker create parakeet-wasm-builder)
docker cp "$CONTAINER_ID:/output/." docs/pkg/
docker rm "$CONTAINER_ID"

echo ""
echo "==> Build complete! Artifacts written to docs/pkg/"
echo "    parakeet_wasm_bg.wasm"
echo "    parakeet_wasm.js"
echo ""
echo "    Start local server:  cd docs && python3 -m http.server 8765"
