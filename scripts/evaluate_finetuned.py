#!/usr/bin/env python3
"""
Evaluate fine-tuned model on all test splits and compare with baseline.
Run inside the NeMo Docker container.
"""
import nemo.collections.asr as nemo_asr
import json
import os
import glob
import time
from nemo.collections.asr.metrics.wer import word_error_rate

DATA_DIR = "/workspace/parakeet-basque/data"
RESULTS_DIR = "/workspace/parakeet-basque/results"
EXP_NAME = "parakeet-tdt-basque"

TEST_SPLITS = ["test_cv", "test_parl", "test_oslr"]

# ============================================================================
# Find the best checkpoint
# ============================================================================
print("=" * 60)
print("Looking for fine-tuned model checkpoint...")
print("=" * 60)

# Search for .nemo files
nemo_files = glob.glob(os.path.join(RESULTS_DIR, EXP_NAME, "**/*.nemo"), recursive=True)
if not nemo_files:
    nemo_files = glob.glob(os.path.join(RESULTS_DIR, "**/*.nemo"), recursive=True)

if not nemo_files:
    print("ERROR: No .nemo checkpoint found!")
    print("Searched in: {}".format(RESULTS_DIR))
    print("Make sure fine-tuning (03_finetune.sh) completed successfully.")
    exit(1)

# Use the most recent .nemo file
nemo_files.sort(key=os.path.getmtime, reverse=True)
model_path = nemo_files[0]
print("Using checkpoint: {}".format(model_path))

# ============================================================================
# Load model
# ============================================================================
print("\nLoading fine-tuned model...")
try:
    asr_model = nemo_asr.models.ASRModel.restore_from(model_path)
except Exception as e:
    print("Error with ASRModel.restore_from: {}".format(e))
    print("Trying EncDecHybridRNNTCTCBPEModel...")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)

asr_model = asr_model.cuda()
asr_model.eval()
print("Model loaded successfully.")

# ============================================================================
# Load baseline results for comparison
# ============================================================================
baseline_summary = {}
baseline_path = os.path.join(RESULTS_DIR, "baseline_summary.json")
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline_summary = json.load(f)
    print("\nLoaded baseline results from: {}".format(baseline_path))

# ============================================================================
# Evaluate on each test split
# ============================================================================
finetuned_summary = {}

for split in TEST_SPLITS:
    manifest_path = os.path.join(DATA_DIR, "{}_manifest.json".format(split))
    if not os.path.exists(manifest_path):
        print("\nWARNING: Manifest not found: {}, skipping.".format(manifest_path))
        continue

    with open(manifest_path) as f:
        num_samples = sum(1 for _ in f)

    print("\n" + "=" * 60)
    print("Evaluating on: {} ({} samples)".format(split, num_samples))
    print("=" * 60)

    output_path = os.path.join(RESULTS_DIR, "finetuned_{}.json".format(split))
    start_time = time.time()

    try:
        # Load references and audio paths from manifest
        references = []
        audio_paths = []
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                references.append(entry["text"])
                audio_paths.append(entry["audio_filepath"])

        # Transcribe using NeMo 2.7 API
        results = asr_model.transcribe(
            use_lhotse=False,
            audio=audio_paths,
            batch_size=16,
            num_workers=4,
            return_hypotheses=False,
            verbose=True,
        )

        # Handle different return formats
        if isinstance(results, tuple):
            predictions = results[0]
        elif hasattr(results, "__iter__") and not isinstance(results, str):
            if len(results) > 0 and hasattr(results[0], "text"):
                predictions = [r.text for r in results]
            else:
                predictions = list(results)
        else:
            predictions = results

        # Compute WER
        wer = word_error_rate(hypotheses=predictions, references=references)
        elapsed = time.time() - start_time

        finetuned_summary[split] = {
            "wer": round(wer * 100, 2),
            "num_samples": num_samples,
            "elapsed_seconds": round(elapsed, 1),
        }

        print("  Fine-tuned WER: {:.2f}%".format(wer * 100))

        # Compare with baseline
        if split in baseline_summary and "wer" in baseline_summary[split]:
            baseline_wer = baseline_summary[split]["wer"]
            improvement = baseline_wer - finetuned_summary[split]["wer"]
            rel_improvement = (improvement / baseline_wer) * 100 if baseline_wer > 0 else 0
            print("  Baseline WER:   {:.2f}%".format(baseline_wer))
            print("  Improvement:    {:+.2f}% absolute ({:+.1f}% relative)".format(improvement, rel_improvement))

        # Save predictions
        with open(output_path, "w") as f:
            for ref, pred, audio in zip(references, predictions, audio_paths):
                entry = {
                    "audio_filepath": audio,
                    "text": ref,
                    "pred_text": pred,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Show examples
        print("\n  Sample predictions:")
        for i in range(min(5, len(predictions))):
            print("    REF:  {}".format(references[i]))
            print("    PRED: {}".format(predictions[i]))
            print()

    except Exception as e:
        print("  ERROR evaluating {}: {}".format(split, e))
        import traceback
        traceback.print_exc()
        finetuned_summary[split] = {"error": str(e)}

# ============================================================================
# Save and display summary
# ============================================================================
summary_path = os.path.join(RESULTS_DIR, "finetuned_summary.json")
with open(summary_path, "w") as f:
    json.dump(finetuned_summary, f, indent=2)

print("\n" + "=" * 60)
print("EVALUATION COMPARISON: BASELINE vs FINE-TUNED")
print("=" * 60)
header = "{:<15} {:<15} {:<18} {:<15}".format("Split", "Baseline WER", "Fine-tuned WER", "Improvement")
print(header)
print("-" * 63)

for split in TEST_SPLITS:
    baseline_wer = baseline_summary.get(split, {}).get("wer", "N/A")
    finetuned_wer = finetuned_summary.get(split, {}).get("wer", "N/A")

    if isinstance(baseline_wer, (int, float)) and isinstance(finetuned_wer, (int, float)):
        imp = baseline_wer - finetuned_wer
        print("{:<15} {:<15.2f} {:<18.2f} {:+.2f}%".format(split, baseline_wer, finetuned_wer, imp))
    else:
        print("{:<15} {:<15} {:<18} {:<15}".format(split, str(baseline_wer), str(finetuned_wer), "N/A"))

print("\nDetailed results saved to: {}".format(summary_path))
print("Model checkpoint: {}".format(model_path))
print("\nNext: Run 05_export.sh to export the model")
