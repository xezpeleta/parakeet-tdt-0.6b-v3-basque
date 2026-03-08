#!/usr/bin/env python3
"""Export fine-tuned Basque Parakeet TDT model to onnx-asr format.

onnx-asr expects (for NeMo TDT models):
  - encoder-model.onnx  (+ encoder-model.onnx.data for large models)
  - decoder_joint-model.onnx
  - vocab.txt
  - config.json

This follows the onnx-asr conversion guide:
  https://istupakov.github.io/onnx-asr/conversion/

Run inside the NeMo Docker container.

Usage with onnx-asr:
  import onnx_asr
  model = onnx_asr.load_model("nemo-conformer-tdt", "path/to/output_dir")
  print(model.recognize("audio.wav"))
"""

import glob
import json
import os
import traceback

MODELS_DIR = "/workspace/parakeet-basque/models"
RESULTS_DIR = "/workspace/parakeet-basque/results"
NEMO_PATH = os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v3-basque.nemo")
OUTPUT_DIR = os.path.join(MODELS_DIR, "onnx-asr-parakeet-tdt-0.6b-v3-basque")


def find_nemo_path():
    """Find the .nemo model file."""
    if os.path.exists(NEMO_PATH):
        return NEMO_PATH
    candidates = glob.glob(os.path.join(RESULTS_DIR, "**", "*.nemo"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .nemo file found in models/ or results/")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Export for onnx-asr (NeMo TDT format)")
    print("=" * 60)

    # ---- Load model ----
    nemo_path = find_nemo_path()
    print("Input .nemo: {}".format(nemo_path))
    print("Output dir:  {}".format(OUTPUT_DIR))

    import nemo.collections.asr as nemo_asr

    print("\nLoading model...")
    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    asr_model.eval()

    # ---- Export ONNX using NeMo's built-in export ----
    # NeMo's export for RNNT/TDT models produces:
    #   encoder-<name>.onnx  (+ external data files)
    #   decoder_joint-<name>.onnx
    onnx_name = "model.onnx"
    onnx_path = os.path.join(OUTPUT_DIR, onnx_name)

    print("\nExporting ONNX via NeMo export()...")
    asr_model.export(onnx_path)

    # NeMo creates encoder-model.onnx and decoder_joint-model.onnx
    encoder_path = os.path.join(OUTPUT_DIR, "encoder-model.onnx")
    decoder_joint_path = os.path.join(OUTPUT_DIR, "decoder_joint-model.onnx")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_joint_path):
        # Try alternate naming patterns
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("encoder-") and f.endswith(".onnx"):
                if f != "encoder-model.onnx":
                    os.rename(
                        os.path.join(OUTPUT_DIR, f),
                        encoder_path,
                    )
            if f.startswith("decoder_joint-") and f.endswith(".onnx"):
                if f != "decoder_joint-model.onnx":
                    os.rename(
                        os.path.join(OUTPUT_DIR, f),
                        decoder_joint_path,
                    )

    # Handle external data files - consolidate them into single .data file
    print("\nConsolidating external data files...")
    _consolidate_encoder(encoder_path)

    # ---- vocab.txt (onnx-asr format: "token id") ----
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i, token in enumerate([*asr_model.tokenizer.vocab, "<blk>"]):
            f.write("{} {}\n".format(token, i))
    print("Saved vocab.txt")

    # ---- config.json ----
    config = {
        "model_type": "nemo-conformer-tdt",
        "features_size": 128,
        "subsampling_factor": 8,
        "max_tokens_per_step": 10,
    }
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    # ---- Clean up stray external weight files ----
    _cleanup_stray_weights(OUTPUT_DIR)

    # ---- Optional: INT8 quantized version ----
    _create_int8_variant(OUTPUT_DIR, encoder_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("onnx-asr export complete!")
    print("Output directory: {}".format(OUTPUT_DIR))
    print("\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print("  {:45s} {:>8.1f} MB".format(f, size_mb))

    print("\nUsage with onnx-asr:")
    print('  import onnx_asr')
    print('  model = onnx_asr.load_model("nemo-conformer-tdt", "{}")'.format(OUTPUT_DIR))
    print('  print(model.recognize("audio.wav"))')
    print()
    print("With INT8 quantization:")
    print('  model = onnx_asr.load_model("nemo-conformer-tdt", "{}", quantization="int8")'.format(OUTPUT_DIR))
    print("=" * 60)


def _consolidate_encoder(encoder_path):
    """Consolidate encoder external data into a single .data file."""
    import onnx

    if not os.path.exists(encoder_path):
        return

    try:
        model = onnx.load(encoder_path)
        output_dir = os.path.dirname(encoder_path)

        # Save with all external data in one file
        onnx.save(
            model,
            encoder_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="encoder-model.onnx.data",
        )
        print("Consolidated encoder external data")
    except Exception as e:
        print("Warning: Could not consolidate encoder data: {}".format(e))


def _cleanup_stray_weights(output_dir):
    """Remove leftover external weight files from NeMo export."""
    keep_patterns = [
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "decoder_joint-model.onnx",
        "vocab.txt",
        "config.json",
        "encoder-model.int8.onnx",
    ]
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if not os.path.isfile(fpath):
            continue
        if f in keep_patterns:
            continue
        # Remove stray weight files (layers.*.weight, onnx__Conv_*, etc.)
        if any(
            pat in f
            for pat in [
                "layers.",
                "onnx__",
                "Constant_",
                "pre_encode.",
                "nemo128.",
            ]
        ):
            os.remove(fpath)
            print("  Cleaned up: {}".format(f))


def _create_int8_variant(output_dir, encoder_path):
    """Create INT8 quantized version of the encoder."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = os.path.join(output_dir, "encoder-model.int8.onnx")
        print("\nCreating INT8 quantized encoder...")
        quantize_dynamic(
            model_input=encoder_path,
            model_output=int8_path,
            weight_type=QuantType.QUInt8,
        )
        size_mb = os.path.getsize(int8_path) / (1024 ** 2)
        print("  INT8 encoder: {:.1f} MB".format(size_mb))
    except ImportError:
        print("\nINFO: onnxruntime not installed, skipping INT8 quantization.")
        print("  Install with: pip install onnxruntime")
    except Exception as e:
        print("\nWarning: INT8 quantization failed: {}".format(e))
        print("  The fp32 ONNX files are still usable.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
