#!/usr/bin/env python3
"""Export fine-tuned Basque Parakeet TDT model to sherpa-onnx format.

sherpa-onnx expects 3 separate ONNX files for transducer models:
  - encoder.onnx  (+ encoder.int8.onnx)
  - decoder.onnx  (+ decoder.int8.onnx)
  - joiner.onnx   (+ joiner.int8.onnx)
  - tokens.txt

This follows the official sherpa-onnx conversion script from:
  https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3

Run inside the NeMo Docker container.

Usage with sherpa-onnx:
  sherpa-onnx-offline \
    --encoder=encoder.int8.onnx \
    --decoder=decoder.int8.onnx \
    --joiner=joiner.int8.onnx \
    --tokens=tokens.txt \
    --model-type=nemo_transducer \
    audio.wav
"""

import glob
import os
import sys
import traceback
from typing import Dict

import onnx
import torch

MODELS_DIR = "/workspace/parakeet-basque/models"
RESULTS_DIR = "/workspace/parakeet-basque/results"
NEMO_PATH = os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v3-basque.nemo")
OUTPUT_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-parakeet-tdt-0.6b-v3-basque")


def find_nemo_path():
    """Find the .nemo model file."""
    if os.path.exists(NEMO_PATH):
        return NEMO_PATH
    candidates = glob.glob(os.path.join(RESULTS_DIR, "**", "*.nemo"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .nemo file found in models/ or results/")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def add_meta_data(filename, meta_data):
    """Add metadata to an ONNX model (in-place)."""
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if "encoder" in os.path.basename(filename) and "int8" not in filename:
        # Save encoder with external data to keep file sizes manageable
        external_filename = os.path.splitext(os.path.basename(filename))[0]
        weights_path = os.path.join(
            os.path.dirname(filename), external_filename + ".weights"
        )
        # Remove existing external data file if present
        if os.path.exists(weights_path):
            os.remove(weights_path)
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


@torch.no_grad()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    print("=" * 60)
    print("Export for sherpa-onnx (transducer: encoder + decoder + joiner)")
    print("=" * 60)

    # ---- Load model ----
    nemo_path = find_nemo_path()
    print("Input .nemo: {}".format(nemo_path))
    print("Output dir:  {}".format(OUTPUT_DIR))

    import nemo.collections.asr as nemo_asr

    print("\nLoading model...")
    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    asr_model.eval()

    # ---- tokens.txt (sherpa-onnx format: "token id") ----
    tokens_path = os.path.join(OUTPUT_DIR, "tokens.txt")
    with open(tokens_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write("{} {}\n".format(s, i))
        f.write("<blk> {}\n".format(i + 1))
    print("Saved tokens.txt ({} tokens)".format(i + 2))

    # ---- Export individual ONNX components ----
    print("\nExporting encoder.onnx ...")
    asr_model.encoder.export(os.path.join(OUTPUT_DIR, "encoder.onnx"))

    print("Exporting decoder.onnx ...")
    asr_model.decoder.export(os.path.join(OUTPUT_DIR, "decoder.onnx"))

    print("Exporting joiner.onnx ...")
    asr_model.joint.export(os.path.join(OUTPUT_DIR, "joiner.onnx"))

    # ---- Build metadata ----
    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "NeMo (fine-tuned for Basque)",
        "url": "https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque",
        "comment": "Basque fine-tuned, transducer branch only",
        "feat_dim": 128,
    }

    # ---- INT8 quantization ----
    print("\nQuantizing to INT8 ...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for m in ["encoder", "decoder", "joiner"]:
            src = os.path.join(OUTPUT_DIR, "{}.onnx".format(m))
            dst = os.path.join(OUTPUT_DIR, "{}.int8.onnx".format(m))
            # Encoder uses QUInt8, decoder/joiner use QInt8
            # (following sherpa-onnx convention)
            wt = QuantType.QUInt8 if m == "encoder" else QuantType.QInt8
            quantize_dynamic(
                model_input=src,
                model_output=dst,
                weight_type=wt,
            )
            size_mb = os.path.getsize(dst) / (1024 ** 2)
            print("  {}: {:.1f} MB".format(dst, size_mb))

        # Add metadata to quantized encoder
        add_meta_data(
            os.path.join(OUTPUT_DIR, "encoder.int8.onnx"), meta_data
        )
        print("Added metadata to encoder.int8.onnx")

    except ImportError:
        print("WARNING: onnxruntime not installed, skipping INT8 quantization.")
        print("  Install with: pip install onnxruntime")
        print("  The fp32 ONNX files are still usable.")

    # Add metadata to fp32 encoder
    add_meta_data(os.path.join(OUTPUT_DIR, "encoder.onnx"), meta_data)
    print("Added metadata to encoder.onnx")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("sherpa-onnx export complete!")
    print("Output directory: {}".format(OUTPUT_DIR))
    print("\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print("  {:40s} {:>8.1f} MB".format(f, size_mb))

    print("\nUsage with sherpa-onnx (int8):")
    print("  sherpa-onnx-offline \\")
    print("    --encoder=encoder.int8.onnx \\")
    print("    --decoder=decoder.int8.onnx \\")
    print("    --joiner=joiner.int8.onnx \\")
    print("    --tokens=tokens.txt \\")
    print("    --model-type=nemo_transducer \\")
    print("    audio.wav")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
