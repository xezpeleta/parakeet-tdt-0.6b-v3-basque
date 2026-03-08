---
language:
- eu
license: cc-by-4.0
pipeline_tag: automatic-speech-recognition
library_name: sherpa-onnx
tags:
- automatic-speech-recognition
- speech
- asr
- basque
- euskara
- onnx
- sherpa-onnx
- parakeet
- fastconformer
- tdt
- int8
- quantized
base_model: itzune/parakeet-tdt-0.6b-v3-basque
datasets:
- asierhv/composite_corpus_eu_v2.1
model-index:
- name: parakeet-tdt-0.6b-v3-basque (sherpa-onnx INT8)
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

# Parakeet TDT 0.6B v3 — Basque (Euskara) · sherpa-onnx INT8

ONNX export of [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) packaged for **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** — a cross-platform, real-time speech recognition engine for edge devices, mobile, embedded systems, and WebAssembly.

The weights are **INT8 dynamically quantised** from the BF16 fine-tuned checkpoint, reducing the total package size from ~2.4 GB to ~641 MB while preserving transcription quality.

---

## Model details

| Property | Value |
|---|---|
| Architecture | FastConformer RNNT-TDT (Parakeet TDT 0.6B v3) |
| Language | Basque (`eu`) |
| Sample rate | 16 kHz mono |
| Parameters | ~600 M |
| Vocabulary size | 1024 tokens (SentencePiece BPE) |
| Quantisation | INT8 dynamic (QUInt8 encoder / QInt8 decoder+joiner) |
| sherpa-onnx model type | `nemo_transducer` |
| Subsampling factor | 8 |
| Feature dimension | 128 log-mel filterbanks |
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Fine-tuned model | [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) |
| Fine-tuning framework | NVIDIA NeMo |
| Hardware | NVIDIA L40 (48 GB) |

---

## Files

| File | Size | Description |
|---|---|---|
| `encoder.int8.onnx` | 623 MB | INT8 encoder (FastConformer) |
| `decoder.int8.onnx` | 12 MB | INT8 prediction network |
| `joiner.int8.onnx` | 6 MB | INT8 joint network |
| `tokens.txt` | 92 KB | Vocabulary (one token per line, index = line number) |

---

## Evaluation

WER measured on held-out test splits from [asierhv/composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1):

| Split | Baseline (base model on Basque) | Fine-tuned |
|---|---:|---:|
| `test_cv` (Common Voice) | 108.47% | **6.92%** |
| `test_parl` (Parliament) | 107.61% | **4.36%** |
| `test_oslr` (OpenSLR) | 108.52% | **14.52%** |

> The base model is English-oriented. WER > 100% on Basque is expected for it; the fine-tuned numbers represent the actual usable quality.

---

## Quick start

### Install sherpa-onnx

```bash
pip install sherpa-onnx
```

Or download a pre-built binary / use the C++ API — see the [sherpa-onnx documentation](https://k2-fsa.github.io/sherpa/onnx/).

### Python API

```python
import sherpa_onnx

# Point to the folder containing the 3 ONNX files + tokens.txt
model_dir = "/path/to/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{model_dir}/encoder.int8.onnx",
    decoder=f"{model_dir}/decoder.int8.onnx",
    joiner=f"{model_dir}/joiner.int8.onnx",
    tokens=f"{model_dir}/tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",
    model_type="nemo_transducer",
)

# Transcribe a WAV file (16 kHz mono)
stream = recognizer.create_stream()
audio, sample_rate = sherpa_onnx.read_wave("/path/to/audio.wav")
stream.accept_waveform(sample_rate, audio)
recognizer.decode_stream(stream)
print(stream.result.text)
```

### Command-line (offline)

```bash
sherpa-onnx \
  --encoder-model=encoder.int8.onnx \
  --decoder-model=decoder.int8.onnx \
  --joiner-model=joiner.int8.onnx \
  --tokens=tokens.txt \
  --decoding-method=greedy_search \
  --model-type=nemo_transducer \
  /path/to/audio.wav
```

### Real-time microphone input (Python)

```python
import sherpa_onnx
import sounddevice as sd
import numpy as np

model_dir = "/path/to/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"

# For streaming/online, use OnlineRecognizer with the same model files
# (sherpa-onnx supports NeMo TDT models both offline and online)
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{model_dir}/encoder.int8.onnx",
    decoder=f"{model_dir}/decoder.int8.onnx",
    joiner=f"{model_dir}/joiner.int8.onnx",
    tokens=f"{model_dir}/tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",
    model_type="nemo_transducer",
    chunk_size=32,
)

stream = recognizer.create_stream()
sample_rate = 16000

def callback(indata, frames, time, status):
    samples = indata[:, 0].astype(np.float32)
    stream.accept_waveform(sample_rate, samples)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result(stream)
    if result:
        print(f"\r{result}", end="", flush=True)

with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
    print("Listening... Press Ctrl+C to stop.")
    import time
    while True:
        time.sleep(0.1)
```

---

## Mobile / embedded / WebAssembly

sherpa-onnx supports many deployment targets beyond Python:

| Platform | Notes |
|---|---|
| Android | Java/Kotlin API, pre-built AAR |
| iOS | Swift/ObjC API |
| Raspberry Pi / ARM | Static C++ binaries available |
| WebAssembly | In-browser speech recognition |
| Windows / macOS / Linux | Native binaries and shared libraries |

See [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) for pre-built packages.

---

## Export recipe

This model was exported from the `.nemo` checkpoint using a custom script based on the [official sherpa-onnx export guide for Parakeet TDT](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3):

1. Load fine-tuned NeMo model
2. Export encoder, decoder, joiner as separate ONNX graphs
3. Add required metadata to each graph (vocab_size, subsampling_factor, feat_dim, etc.)
4. Apply INT8 dynamic quantisation via `onnxruntime.quantization.quantize_dynamic`

The export and fine-tuning code is available at: [xezpeleta/parakeet-tdt-0.6b-v3-basque](https://github.com/xezpeleta/parakeet-tdt-0.6b-v3-basque).

---

## Related models

| Repo | Format | Use case |
|---|---|---|
| [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) | NeMo `.nemo` | Full NeMo / PyTorch inference & fine-tuning |
| [xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr) | ONNX-ASR | Simple Python inference via `onnx-asr` |
| **This repo** | sherpa-onnx INT8 | On-device / real-time / cross-platform |

---

## Citation and acknowledgements

If you use this model, please credit:

1. Base model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
2. Fine-tuned model: [itzune/parakeet-tdt-0.6b-v3-basque](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque)
3. Training dataset: [asierhv/composite_corpus_eu_v2.1](https://huggingface.co/datasets/asierhv/composite_corpus_eu_v2.1)
4. sherpa-onnx: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

Underlying source collections in the training corpus:
- Mozilla Common Voice (Basque)
- Basque Parliament corpus
- OpenSLR Basque resources

## License

CC BY 4.0. Inherit license obligations from the base model and dataset.
