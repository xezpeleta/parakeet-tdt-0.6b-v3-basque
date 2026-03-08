#!/usr/bin/env python3
"""
Export fine-tuned model and generate model card.
Run inside the NeMo Docker container.
"""
import os
import json
import glob
import shutil

RESULTS_DIR = "/workspace/parakeet-basque/results"
EXPORT_DIR = "/workspace/parakeet-basque/models"
EXP_NAME = "parakeet-tdt-basque"

os.makedirs(EXPORT_DIR, exist_ok=True)

# ============================================================================
# Find best checkpoint
# ============================================================================
print("=" * 60)
print("Exporting fine-tuned model")
print("=" * 60)

nemo_files = glob.glob(os.path.join(RESULTS_DIR, "**/*.nemo"), recursive=True)
if not nemo_files:
    print("ERROR: No .nemo checkpoint found!")
    exit(1)

nemo_files.sort(key=os.path.getmtime, reverse=True)
best_model = nemo_files[0]
print("Best checkpoint: {}".format(best_model))

# Copy to export directory
export_path = os.path.join(EXPORT_DIR, "parakeet-tdt-0.6b-v3-basque.nemo")
shutil.copy2(best_model, export_path)
print("Exported to: {}".format(export_path))
print("Size: {:.2f} GB".format(os.path.getsize(export_path) / (1024**3)))

# ============================================================================
# Generate model card
# ============================================================================
# Load evaluation results
baseline_summary = {}
finetuned_summary = {}

baseline_path = os.path.join(RESULTS_DIR, "baseline_summary.json")
finetuned_path = os.path.join(RESULTS_DIR, "finetuned_summary.json")

if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline_summary = json.load(f)

if os.path.exists(finetuned_path):
    with open(finetuned_path) as f:
        finetuned_summary = json.load(f)

# Build WER comparison table
wer_table = "| Test Split | Baseline WER | Fine-tuned WER | Improvement |\n"
wer_table += "|---|---|---|---|\n"
for split in ["test_cv", "test_parl", "test_oslr"]:
    b = baseline_summary.get(split, {}).get("wer", "N/A")
    ft = finetuned_summary.get(split, {}).get("wer", "N/A")
    if isinstance(b, (int, float)) and isinstance(ft, (int, float)):
        imp = "{:+.2f}%".format(b - ft)
    else:
        imp = "N/A"
    wer_table += "| {} | {}% | {}% | {} |\n".format(split, b, ft, imp)

model_card = """---
language:
- eu
license: cc-by-4.0
tags:
- automatic-speech-recognition
- nemo
- basque
- parakeet
- fastconformer
- tdt
datasets:
- asierhv/composite_corpus_eu_v2.1
base_model: nvidia/parakeet-tdt-0.6b-v3
pipeline_tag: automatic-speech-recognition
---

# Parakeet TDT 0.6B v3 — Basque Fine-tuned

This model is a fine-tuned version of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) on Basque (Euskara) speech data.

## Model Description

- **Base model**: nvidia/parakeet-tdt-0.6b-v3 (600M parameters, FastConformer-TDT)
- **Language**: Basque (eu)
- **Training data**: [asierhv/composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1) (~676 hours)
  - Mozilla Common Voice 18.0 (Basque)
  - Basque Parliament 1
  - OpenSLR 76 (Basque)
- **Fine-tuning**: Full model fine-tuning with AdamW optimizer, CosineAnnealing scheduler
- **Hardware**: NVIDIA L40 48GB

## Evaluation Results

{}

## Usage

```python
import nemo.collections.asr as nemo_asr

# Load fine-tuned model
asr_model = nemo_asr.models.ASRModel.restore_from("parakeet-tdt-0.6b-v3-basque.nemo")

# Transcribe
output = asr_model.transcribe(["path/to/basque_audio.wav"])
print(output[0].text)
```

## Training Details

- Learning rate: 1e-4
- Scheduler: CosineAnnealing with 1000 warmup steps
- Effective batch size: 64 (8 x 8 gradient accumulation)
- Precision: BF16 mixed
- Framework: NVIDIA NeMo

## License

This model is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), same as the base model.
""".format(wer_table)

model_card_path = os.path.join(EXPORT_DIR, "README.md")
with open(model_card_path, "w") as f:
    f.write(model_card)
print("Model card saved to: {}".format(model_card_path))

# ============================================================================
# Optional: Push to HuggingFace
# ============================================================================
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token and hf_token != "your_huggingface_token_here":
    print("\nHF_TOKEN found. To push to HuggingFace, run:")
    print("  huggingface-cli upload xezpeleta/parakeet-tdt-0.6b-v3-basque {}".format(EXPORT_DIR))
    print("  Or use the HuggingFace web interface to upload the .nemo file and README.md")
else:
    print("\nNo HF_TOKEN configured. Skipping HuggingFace upload.")
    print("To upload manually:")
    print("  1. Copy {} and {} to your local machine".format(export_path, model_card_path))
    print("  2. Upload to HuggingFace via web interface or CLI")

print("\n" + "=" * 60)
print("Export complete!")
print("  Model: {}".format(export_path))
print("  Card:  {}".format(model_card_path))
print("=" * 60)
