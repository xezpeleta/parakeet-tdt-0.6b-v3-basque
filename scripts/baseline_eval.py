#!/usr/bin/env python3
"""
Baseline evaluation: measure WER of pretrained model on Basque test sets.
Run inside the NeMo Docker container.
"""
import nemo.collections.asr as nemo_asr
import json
import os
import time
from nemo.collections.asr.metrics.wer import word_error_rate

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DATA_DIR = "/workspace/parakeet-basque/data"
RESULTS_DIR = "/workspace/parakeet-basque/results"

TEST_SPLITS = ["test_cv", "test_parl", "test_oslr"]

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Loading pretrained model: {}".format(MODEL_NAME))
print("=" * 60)

try:
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    print("Model loaded successfully. Type: {}".format(type(asr_model).__name__))
except Exception as e:
    print("ERROR loading model: {}".format(e))
    print("Trying alternative loading method...")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=MODEL_NAME)

# Move to GPU
asr_model = asr_model.cuda()
asr_model.eval()

results_summary = {}

for split in TEST_SPLITS:
    manifest_path = os.path.join(DATA_DIR, "{}_manifest.json".format(split))
    if not os.path.exists(manifest_path):
        print("\nWARNING: Manifest not found: {}, skipping.".format(manifest_path))
        continue

    # Count samples
    with open(manifest_path) as f:
        num_samples = sum(1 for _ in f)

    print("\n" + "=" * 60)
    print("Evaluating on: {} ({} samples)".format(split, num_samples))
    print("=" * 60)

    output_path = os.path.join(RESULTS_DIR, "baseline_{}.json".format(split))
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
            use_lhotse=False, # Old CUDA compatibility issue workaround
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
        results_summary[split] = {
            "wer": round(wer * 100, 2),
            "num_samples": num_samples,
            "elapsed_seconds": round(elapsed, 1),
        }

        print("  WER: {:.2f}%".format(wer * 100))
        print("  Time: {:.1f}s".format(elapsed))

        # Save detailed predictions
        with open(output_path, "w") as f:
            for ref, pred, audio in zip(references, predictions, audio_paths):
                entry = {
                    "audio_filepath": audio,
                    "text": ref,
                    "pred_text": pred,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("  Predictions saved to: {}".format(output_path))

        # Show some examples
        print("\n  Sample predictions:")
        for i in range(min(5, len(predictions))):
            print("    REF:  {}".format(references[i]))
            print("    PRED: {}".format(predictions[i]))
            print()

    except Exception as e:
        print("  ERROR evaluating {}: {}".format(split, e))
        import traceback
        traceback.print_exc()
        results_summary[split] = {"error": str(e)}

# Save summary
summary_path = os.path.join(RESULTS_DIR, "baseline_summary.json")
with open(summary_path, "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 60)
print("BASELINE EVALUATION SUMMARY")
print("=" * 60)
for split, result in results_summary.items():
    if "error" in result:
        print("  {}: ERROR - {}".format(split, result["error"]))
    else:
        print("  {}: WER = {}% ({} samples, {}s)".format(
            split, result["wer"], result["num_samples"], result["elapsed_seconds"]))
print("\nSummary saved to: {}".format(summary_path))
print("\nNext: Run 03_finetune.sh")
