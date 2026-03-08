#!/usr/bin/env python3
"""Export fine-tuned Basque Parakeet model to ONNX.

Run inside the NeMo Docker container.
"""

import glob
import os
import traceback

import nemo.collections.asr as nemo_asr

RESULTS_DIR = "/workspace/parakeet-basque/results"
EXPORT_DIR = "/workspace/parakeet-basque/models"
DEFAULT_NEMO = os.path.join(EXPORT_DIR, "parakeet-tdt-0.6b-v3-basque.nemo")
ONNX_PATH = os.path.join(EXPORT_DIR, "parakeet-tdt-0.6b-v3-basque.onnx")


def find_nemo_path() -> str:
    if os.path.exists(DEFAULT_NEMO):
        return DEFAULT_NEMO

    candidates = glob.glob(os.path.join(RESULTS_DIR, "**", "*.nemo"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .nemo file found in models/ or results/")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def try_export(model, output_path: str) -> None:
    errors = []

    attempts = [
        {"output": output_path, "check_trace": False},
        {"output": output_path},
        {"filename": output_path, "check_trace": False},
        {"filename": output_path},
    ]

    for kwargs in attempts:
        try:
            print("Trying export with kwargs={}".format(kwargs))
            model.export(**kwargs)
            return
        except Exception as exc:
            errors.append((kwargs, exc))

    print("All export attempts failed:\n")
    for kwargs, exc in errors:
        print("- kwargs={}: {}".format(kwargs, exc))
    raise RuntimeError("ONNX export failed for all known NeMo export signatures")


def main() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("=" * 60)
    print("Exporting ONNX")
    print("=" * 60)

    nemo_path = find_nemo_path()
    print("Input .nemo: {}".format(nemo_path))
    print("Output .onnx: {}".format(ONNX_PATH))

    print("\nLoading model...")
    model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    model.eval()

    print("\nRunning ONNX export...")
    try_export(model, ONNX_PATH)

    base_name = os.path.basename(ONNX_PATH)
    encoder_path = os.path.join(EXPORT_DIR, "encoder-{}".format(base_name))
    decoder_joint_path = os.path.join(EXPORT_DIR, "decoder_joint-{}".format(base_name))

    single_exists = os.path.exists(ONNX_PATH)
    split_exists = os.path.exists(encoder_path) and os.path.exists(decoder_joint_path)

    if not single_exists and not split_exists:
        raise RuntimeError(
            "Export call returned, but neither a single ONNX file nor RNNT split ONNX files were created"
        )

    print("\nExport successful!")
    if single_exists:
        size_gb = os.path.getsize(ONNX_PATH) / (1024 ** 3)
        print("ONNX path: {}".format(ONNX_PATH))
        print("Size: {:.2f} GB".format(size_gb))

    if split_exists:
        enc_size_gb = os.path.getsize(encoder_path) / (1024 ** 3)
        dec_size_gb = os.path.getsize(decoder_joint_path) / (1024 ** 3)
        print("RNNT split ONNX export detected:")
        print("  Encoder: {} ({:.2f} GB)".format(encoder_path, enc_size_gb))
        print("  Decoder+Joint: {} ({:.2f} GB)".format(decoder_joint_path, dec_size_gb))

    metadata_path = os.path.join(EXPORT_DIR, "onnx_export_files.txt")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        if single_exists:
            handle.write(ONNX_PATH + "\n")
        if split_exists:
            handle.write(encoder_path + "\n")
            handle.write(decoder_joint_path + "\n")
    print("File list saved to: {}".format(metadata_path))
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
