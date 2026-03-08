---
language:
- eu
license: cc-by-4.0
pipeline_tag: automatic-speech-recognition
library_name: onnx-asr
tags:
- automatic-speech-recognition
- speech
- asr
- basque
- euskara
- onnx
- onnx-asr
- parakeet
- fastconformer
- tdt
- int8
- quantized
base_model: itzune/parakeet-tdt-0.6b-v3-basque
datasets:
- asierhv/composite_corpus_eu_v2.1
model-index:
- name: parakeet-tdt-0.6b-v3-basque (onnx-asr INT8)
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      type: custom
      name: Composite Basque test splits (CV/Parliament/OSLR)
    metrics:
    - type: wer
      name: test_cv WER
      value: 6.92
    - type: wer
      name: test_parl WER
      value: 4.36
    - type: wer
      name: test_oslr WER
      value: 14.52
---

# Parakeet TDT 0.6B v3 — Basque (Euskara) · ONNX-ASR

ONNX export of [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) packaged for **[onnx-asr](https://github.com/istupakov/onnx-asr)** — a lightweight, pure-Python speech recognition library that runs entirely on ONNX Runtime, no PyTorch or NeMo required.

The encoder is **INT8 dynamically quantised**, reducing its size from ~2.3 GB to ~623 MB.

---

## Model details

| Property | Value |
|---|---|
| Architecture | FastConformer RNNT-TDT (Parakeet TDT 0.6B v3) |
| Language | Basque (`eu`) |
| Sample rate | 16 kHz mono |
| Parameters | ~600 M |
| Vocabulary size | 1024 tokens (SentencePiece BPE) |
| Quantisation | INT8 dynamic encoder (`encoder-model.int8.onnx`) |
| onnx-asr model type | `nemo-conformer-tdt` |
| Features size | 128 log-mel filterbanks |
| Subsampling factor | 8 |
| Max tokens per step | 10 |
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Fine-tuned model | [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) |
| Fine-tuning framework | NVIDIA NeMo |
| Hardware | NVIDIA L40 (48 GB) |

---

## Files

| File | Size | Description |
|---|---|---|
| `encoder-model.int8.onnx` | 623 MB | INT8 encoder (FastConformer, self-contained) |
| `decoder_joint-model.onnx` | 70 MB | Decoder + joint network (FP32) |
| `vocab.txt` | 92 KB | Vocabulary (one token per line) |
| `config.json` | — | onnx-asr model configuration |

`config.json` contents:

```json
{
  "model_type": "nemo-conformer-tdt",
  "features_size": 128,
  "subsampling_factor": 8,
  "max_tokens_per_step": 10
}
```

---

## Evaluation

WER measured on held-out test splits from [asierhv/composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1):

| Split | Baseline (base model on Basque) | Fine-tuned |
|---|---:|---:|
| `test_cv` (Common Voice) | 108.47% | **6.92%** |
| `test_parl` (Parliament) | 107.61% | **4.36%** |
| `test_oslr` (OpenSLR) | 108.52% | **14.52%** |

> The base model is English-oriented. WER > 100% on Basque is expected for it.

---

## Quick start

### Install onnx-asr

```bash
pip install onnx-asr
```

### Transcribe from Hugging Face (automatic download)

```python
import onnx_asr

# Load directly from this HF repo — files are downloaded automatically
model = onnx_asr.load_model("xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr")

text = model.transcribe("/path/to/audio.wav")
print(text)
```

### Transcribe a local folder of files

```python
import onnx_asr
from pathlib import Path

model = onnx_asr.load_model("xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr")

audio_files = list(Path("/path/to/wavs").glob("*.wav"))
for audio_path in audio_files:
    text = model.transcribe(str(audio_path))
    print(f"{audio_path.name}: {text}")
```

### Load from local directory

```python
import onnx_asr

model = onnx_asr.load_model("/path/to/parakeet-tdt-0.6b-v3-basque-onnx-asr")
text = model.transcribe("/path/to/audio.wav")
print(text)
```

### Batch transcription

```python
import onnx_asr

model = onnx_asr.load_model("xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr")

audio_paths = [
    "/path/a.wav",
    "/path/b.wav",
    "/path/c.wav",
]

results = model.transcribe_batch(audio_paths)
for path, text in zip(audio_paths, results):
    print(f"{path}: {text}")
```

### Use FP32 encoder instead of INT8

The INT8 encoder is used by default (smaller, faster). If you want the full-precision encoder, pass it explicitly:

```python
import onnx_asr

model = onnx_asr.load_model(
    "xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr",
    encoder="encoder-model.onnx",  # FP32 variant
)
```

### Use GPU (CUDA) with ONNX Runtime

```bash
pip install onnxruntime-gpu
```

```python
import onnx_asr

model = onnx_asr.load_model(
    "xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

---

## Audio requirements

- **Sample rate**: 16 kHz
- **Channels**: mono (single channel)
- **Format**: WAV, FLAC, or any format readable by `soundfile`/`librosa`

---

## When to choose this format vs. sherpa-onnx

| | onnx-asr (this repo) | sherpa-onnx |
|---|---|---|
| Install | `pip install onnx-asr` | `pip install sherpa-onnx` |
| API | Simple `model.transcribe()` | More control (streaming, chunk size) |
| Streaming | No | Yes |
| Mobile / embedded | No | Yes (Android, iOS, WASM) |
| C++ / native binary | No | Yes |
| Best for | Server-side Python batch transcription | Edge, real-time, multi-platform |

For the sherpa-onnx version of this model, see: [xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx)

---

## Export recipe

This model was exported from the `.nemo` checkpoint using a custom script:

1. Load fine-tuned NeMo model
2. Use NeMo's built-in `asr_model.export("model.onnx")` to produce split graphs
3. Consolidate external data tensors into self-contained ONNX files
4. Generate `vocab.txt` from the SentencePiece tokenizer
5. Write `config.json` with `model_type: nemo-conformer-tdt`
6. Apply INT8 dynamic quantisation to the encoder via `onnxruntime.quantization.quantize_dynamic`

The export and fine-tuning code is available at: [xezpeleta/parakeet-tdt-0.6b-v3-basque](https://github.com/xezpeleta/parakeet-tdt-0.6b-v3-basque).

---

## Related models

| Repo | Format | Use case |
|---|---|---|
| [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) | NeMo `.nemo` | Full NeMo / PyTorch inference & fine-tuning |
| [xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx) | sherpa-onnx INT8 | On-device / real-time / cross-platform |
| **This repo** | ONNX-ASR | Simple Python batch inference |

---

## Citation and acknowledgements

If you use this model, please credit:

1. Base model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
2. Fine-tuned model: [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque)
3. Training dataset: [asierhv/composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1)
4. onnx-asr: [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr)

Underlying source collections in the training corpus:
- Mozilla Common Voice (Basque)
- Basque Parliament corpus
- OpenSLR Basque resources

## License

CC BY 4.0. Inherit license obligations from the base model and dataset.
