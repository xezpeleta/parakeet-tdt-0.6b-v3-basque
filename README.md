# Basque ASR Fine-tuning Toolkit (Parakeet TDT 0.6B v3)

End-to-end repo to fine-tune NVIDIA Parakeet on Basque speech, evaluate results, export a `.nemo` model, and run inference scripts.

This repository complements the published Hugging Face models:

| Repo | Format | Use case |
|---|---|---|
| [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) | NeMo `.nemo` | Full NeMo / PyTorch inference & fine-tuning |
| [xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr) | ONNX-ASR INT8 | Simple Python batch inference (`pip install onnx-asr`) |
| [xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx) | sherpa-onnx INT8 | On-device / real-time / cross-platform |

- **Base model**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- **Training dataset**: https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1

---

## What this repo contains

- Docker-first training/evaluation pipeline (NeMo container on GPU server)
- Dataset download + manifest generation for Basque corpus
- Baseline and fine-tuned evaluation scripts
- Model export script (final `.nemo` + model card)
- Local inference scripts (offline and streaming-style)

---

## Repository structure

```text
config/
  server.conf                   # Server/container configuration

docs/
  ## Plan: Fine-tune Parakeet TDT 0.prompt.md

scripts/
  00_setup.sh                   # Start container + install Python deps
  01_download_dataset.sh        # Download + prepare dataset/manifests
  01b_create_tarred_dataset.sh  # Optional tarred dataset creation
  02_baseline_eval.sh           # Baseline WER on test splits
  03_finetune.sh                # Fine-tuning
  04_evaluate.sh                # Fine-tuned evaluation
  05_export.sh                  # Export final model/model card
  06_export_onnx.sh             # Export basic NeMo ONNX artifacts
  07_export_optimized_onnx.sh   # Export sherpa-onnx + onnx-asr INT8 packages

  run_all.sh                    # Pipeline runner (all steps)
  deploy.sh                     # Deploy scripts to server

  prepare_dataset.py
  baseline_eval.py
  finetune.py
  evaluate_finetuned.py
  export_model.py
  export_onnx.py
  export_sherpa_onnx.py         # Export → sherpa-onnx (encoder/decoder/joiner INT8)
  export_onnx_asr.py            # Export → onnx-asr (encoder+decoder_joint INT8)
  08_upload_onnx_hf.py          # Upload both ONNX packages to Hugging Face

  inference.py                  # Offline/batch inference from HF/local model
  streaming_inference.py        # Streaming-style incremental inference

  hf_cards/                     # Model card READMEs for HF repos
    sherpa-onnx-README.md
    onnx-asr-README.md
```

---

## Training pipeline (Docker on GPU server)

### 1) Configure server values

Edit `config/server.conf` with your server/container values.

### 2) Add secrets

Create `.env` from `.env.example` and set:

- `HF_TOKEN`
- `WANDB_API_KEY` (optional but recommended)
- `WANDB_PROJECT`
- `WANDB_ENTITY` (optional)

### 3) Run step-by-step

```bash
bash scripts/00_setup.sh
bash scripts/01_download_dataset.sh
bash scripts/02_baseline_eval.sh
bash scripts/03_finetune.sh
bash scripts/04_evaluate.sh
bash scripts/05_export.sh
```

Optional tarred dataset step:

```bash
bash scripts/01b_create_tarred_dataset.sh
```

### 4) Or run all

```bash
bash scripts/run_all.sh
```

Start from a specific step index:

```bash
bash scripts/run_all.sh 3
```

### 5) Export optimised ONNX artifacts

```bash
bash scripts/07_export_optimized_onnx.sh
```

This exports two INT8-quantised ONNX packages for different deployment targets:

- `models/sherpa-onnx-parakeet-tdt-0.6b-v3-basque/`
  — INT8 encoder / decoder / joiner split for **sherpa-onnx** (edge, mobile, WASM, real-time)
  — `encoder.int8.onnx` (623 MB), `decoder.int8.onnx` (12 MB), `joiner.int8.onnx` (6 MB), `tokens.txt`

- `models/onnx-asr-parakeet-tdt-0.6b-v3-basque/`
  — INT8 encoder + FP32 decoder/joiner for **onnx-asr** (`pip install onnx-asr`, simple Python)
  — `encoder-model.int8.onnx` (623 MB), `decoder_joint-model.onnx` (70 MB), `vocab.txt`, `config.json`

To upload both packages to Hugging Face:

```bash
python scripts/08_upload_onnx_hf.py
```

> The original basic ONNX export (`06_export_onnx.sh`) is still available but produces raw NeMo split graphs with many external weight files.

---

## Inference usage (with `uv`)

The inference scripts in this repo can run independently of the training pipeline.

### 1) Create environment with `uv`

```bash
cd /path/to/parakeet-tdt-0.6b-v3-basque
uv venv
source .venv/bin/activate
```

### 2) Install dependencies

CPU install:

```bash
uv pip install nemo_toolkit[asr] soundfile librosa
```

If you want GPU PyTorch, install Torch first for your CUDA version, then NeMo:

```bash
# Example (adjust index/versions to your CUDA + driver)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install nemo_toolkit[asr] soundfile librosa
```

### 3) Offline/batch inference

Single file:

```bash
python scripts/inference.py --audio /path/to/audio.wav
```

Multiple files:

```bash
python scripts/inference.py --audio a.wav b.wav c.wav --batch-size 8
```

Directory + JSONL output:

```bash
python scripts/inference.py \
  --audio-dir /path/to/wavs \
  --pattern "*.wav" \
  --output predictions.jsonl
```

Use a specific model path/repo:

```bash
python scripts/inference.py --model itzune/parakeet-tdt-0.6b-v3-basque --audio sample.wav
```

### 4) Streaming-style inference

This simulates online ASR by chunking local audio and printing stable/partial updates.

```bash
python scripts/streaming_inference.py --audio /path/to/audio.wav
```

Tune chunking behavior:

```bash
python scripts/streaming_inference.py \
  --audio /path/to/audio.wav \
  --chunk-seconds 2.0 \
  --stride-seconds 0.5 \
  --context-seconds 8.0
```

Mimic real-time:

```bash
python scripts/streaming_inference.py --audio /path/to/audio.wav --realtime
```

---

## Evaluation outputs

Main output files are written under `results/` on the server workspace:

- `baseline_summary.json`
- `finetuned_summary.json`
- `baseline_<split>.json`
- `finetuned_<split>.json`

Model export locations:

- `models/parakeet-tdt-0.6b-v3-basque.nemo`
- `models/README.md` (model card for HF)
- `models/sherpa-onnx-parakeet-tdt-0.6b-v3-basque/` (sherpa-onnx INT8 bundle)
- `models/onnx-asr-parakeet-tdt-0.6b-v3-basque/` (onnx-asr INT8 bundle)
- `models/onnx-parakeet-tdt-0.6b-v3-basque/` (raw NeMo ONNX split graphs)

---

## Notes and troubleshooting

- NeMo 2.7 may require `use_lhotse=False` in `transcribe(...)` depending on environment behavior.
- If running in containers and packages disappear after restart, reinstall Python dependencies in the active container.
- For GPU issues, verify host driver/container CUDA compatibility and check `torch.cuda.is_available()` inside the runtime.

---

## Credits

- NVIDIA NeMo + Parakeet base model
- Basque composite dataset and its source collections:
  - Mozilla Common Voice
  - Basque Parliament corpus
  - OpenSLR Basque resources

If you use this work, please cite the base model and dataset sources linked above.
