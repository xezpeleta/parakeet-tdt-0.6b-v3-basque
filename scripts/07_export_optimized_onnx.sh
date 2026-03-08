#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Export optimized ONNX packages for lightweight speech-to-text inference.
#
# Creates TWO export formats:
#
# 1. sherpa-onnx format  -> For edge/embedded deployment (Android, iOS, WASM,
#                           Raspberry Pi, etc.) via C++/Python/Go/etc. SDK
#                           https://github.com/k2-fsa/sherpa-onnx
#
# 2. onnx-asr format     -> For Python server-side usage with simple API
#                           (pip install onnx-asr), batch processing, Gradio
#                           https://github.com/istupakov/onnx-asr
#
# Both include INT8 quantized variants for smaller size & faster CPU inference.
# =============================================================================

CONTAINER_NAME="parakeet-basque"

echo "============================================================"
echo " Export Optimized ONNX Packages"
echo "============================================================"
echo ""

# Ensure onnxruntime is installed for quantization
echo ">> Ensuring onnxruntime is installed..."
docker exec "${CONTAINER_NAME}" pip install -q onnxruntime 2>&1 | tail -1

echo ""

# ---- Step 1: sherpa-onnx export ----
echo "============================================================"
echo " [1/2] Exporting for sherpa-onnx"
echo "       (encoder.onnx + decoder.onnx + joiner.onnx + tokens.txt)"
echo "============================================================"
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/export_sherpa_onnx.py
echo ""

# ---- Step 2: onnx-asr export ----
echo "============================================================"
echo " [2/2] Exporting for onnx-asr"
echo "       (encoder-model.onnx + decoder_joint-model.onnx + vocab.txt)"
echo "============================================================"
docker exec "${CONTAINER_NAME}" python /workspace/parakeet-basque/scripts/export_onnx_asr.py
echo ""

# ---- Summary ----
echo "============================================================"
echo " Export Complete!"
echo "============================================================"
echo ""
echo "Output directories on server:"
echo "  sherpa-onnx: ${HOME}/parakeet-basque/models/sherpa-onnx-parakeet-tdt-0.6b-v3-basque/"
echo "  onnx-asr:    ${HOME}/parakeet-basque/models/onnx-asr-parakeet-tdt-0.6b-v3-basque/"
echo ""
echo "--- sherpa-onnx usage (edge/embedded, C++/Python/Go/etc.) ---"
echo "  # Install: pip install sherpa-onnx"
echo "  sherpa-onnx-offline \\"
echo "    --encoder=encoder.int8.onnx \\"
echo "    --decoder=decoder.int8.onnx \\"
echo "    --joiner=joiner.int8.onnx \\"
echo "    --tokens=tokens.txt \\"
echo "    --model-type=nemo_transducer \\"
echo "    audio.wav"
echo ""
echo "--- onnx-asr usage (Python server-side, simple API) ---"
echo "  # Install: pip install onnx-asr[cpu,hub]"
echo "  import onnx_asr"
echo "  model = onnx_asr.load_model('nemo-conformer-tdt', 'path/to/onnx-asr-parakeet-tdt-0.6b-v3-basque')"
echo "  print(model.recognize('audio.wav'))"
echo ""
