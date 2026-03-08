#!/usr/bin/env python3
"""
Download asierhv/composite_corpus_eu_v2.1 dataset and create NeMo manifests.
Run inside the NeMo Docker container.
"""
import os
import json
import soundfile as sf
import numpy as np
from datasets import load_dataset
import librosa
import time

WORK_DIR = "/workspace/parakeet-basque/data"
SAMPLE_RATE = 16000

# Splits to process
SPLITS = {
    "train": "train",
    "dev": "dev",
    "test_cv": "test_cv",
    "test_parl": "test_parl",
    "test_oslr": "test_oslr",
}


def process_split(ds, split_name, output_dir):
    """Process a dataset split: export audio as WAV and create NeMo manifest."""
    audio_dir = os.path.join(output_dir, split_name)
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, split_name + "_manifest.json")

    print("\nProcessing split: {} ({} samples)".format(split_name, len(ds)))
    start_time = time.time()

    skipped = 0
    written = 0

    with open(manifest_path, "w") as manifest_file:
        for idx, sample in enumerate(ds):
            try:
                # Get audio data
                audio = sample["audio"]
                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]

                # Get text
                text = sample.get("sentence", sample.get("text", ""))
                if not text or not isinstance(text, str) or text.strip() == "":
                    skipped += 1
                    continue

                text = text.strip()

                # Resample to 16kHz if needed
                if sr != SAMPLE_RATE:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

                # Save as WAV
                wav_filename = "{}_{:07d}.wav".format(split_name, idx)
                wav_path = os.path.join(audio_dir, wav_filename)
                sf.write(wav_path, audio_array, SAMPLE_RATE)

                # Calculate duration
                duration = len(audio_array) / SAMPLE_RATE

                # Skip very short or very long audio
                if duration < 0.5 or duration > 30.0:
                    skipped += 1
                    continue

                # Write manifest entry
                entry = {
                    "audio_filepath": wav_path,
                    "text": text,
                    "duration": round(duration, 3),
                }
                manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

                if (idx + 1) % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    print("  [{}] {}/{} ({:.0f} samples/s) - written: {}, skipped: {}".format(
                        split_name, idx + 1, len(ds), rate, written, skipped))

            except Exception as e:
                print("  Error processing sample {}: {}".format(idx, e))
                skipped += 1
                continue

    elapsed = time.time() - start_time
    print("  [{}] Done: {} samples written, {} skipped ({:.1f}s)".format(
        split_name, written, skipped, elapsed))
    print("  Manifest: {}".format(manifest_path))
    return manifest_path


def analyze_charset(manifest_path):
    """Analyze character set in the manifest to check tokenizer coverage."""
    charset = {}
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            for ch in entry["text"]:
                charset[ch] = charset.get(ch, 0) + 1

    print("\n  Character set analysis for {}:".format(os.path.basename(manifest_path)))
    print("  Total unique characters: {}".format(len(charset)))
    # Sort by frequency
    sorted_chars = sorted(charset.items(), key=lambda x: -x[1])
    top30 = " ".join(repr(c) for c, _ in sorted_chars[:30])
    print("  Top 30: {}".format(top30))
    # Show rare characters (< 10 occurrences)
    rare = [(c, n) for c, n in sorted_chars if n < 10]
    if rare:
        print("  Rare chars (<10 occurrences): {}".format(rare))
    return charset


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading dataset: asierhv/composite_corpus_eu_v2.1")
    print("=" * 60)

    # Load dataset - this will download and cache it
    ds = load_dataset("asierhv/composite_corpus_eu_v2.1")
    print("\nAvailable splits: {}".format(list(ds.keys())))

    # Process each split
    manifests = {}
    for split_key, split_name in SPLITS.items():
        if split_key in ds:
            manifests[split_name] = process_split(ds[split_key], split_name, WORK_DIR)
        else:
            print("WARNING: Split '{}' not found in dataset. Available: {}".format(
                split_key, list(ds.keys())))

    # Character set analysis on train manifest
    if "train" in manifests:
        analyze_charset(manifests["train"])

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    for split_name, path in manifests.items():
        # Count lines in manifest
        with open(path) as f:
            count = sum(1 for _ in f)
        print("  {}: {} samples -> {}".format(split_name, count, path))
    print("\nNext: Run 02_baseline_eval.sh")


if __name__ == "__main__":
    main()
