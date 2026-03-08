#!/usr/bin/env python3
"""Streaming-style ASR inference over a local WAV file.

This script simulates online decoding by processing overlapping chunks from an
input audio file and printing incremental transcript updates.

Example:
  python scripts/streaming_inference.py --audio /path/to/audio.wav
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr
import soundfile as sf

DEFAULT_MODEL = "itzune/parakeet-tdt-0.6b-v3-basque"
TARGET_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming-style inference for Basque Parakeet model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model repo or local .nemo path")
    parser.add_argument("--audio", required=True, help="Input WAV file (16 kHz recommended)")
    parser.add_argument("--chunk-seconds", type=float, default=2.0, help="Chunk duration in seconds")
    parser.add_argument("--stride-seconds", type=float, default=0.5, help="Stride between chunk starts")
    parser.add_argument("--context-seconds", type=float, default=8.0, help="Audio context sent at each step")
    parser.add_argument("--realtime", action="store_true", help="Sleep to mimic realtime processing")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    parser.add_argument("--use-lhotse", action="store_true", help="Enable NeMo lhotse dataloader")
    return parser.parse_args()


def longest_common_prefix_words(left: str, right: str) -> str:
    left_words = left.strip().split()
    right_words = right.strip().split()
    common = []
    for first, second in zip(left_words, right_words):
        if first != second:
            break
        common.append(first)
    return " ".join(common)


def normalize_prediction(prediction) -> str:
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    if isinstance(prediction, list) and prediction:
        item = prediction[0]
        return item.text if hasattr(item, "text") else str(item)
    return str(prediction)


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    audio, sample_rate = sf.read(str(audio_path))
    if sample_rate != TARGET_SAMPLE_RATE:
        raise SystemExit(
            f"Unsupported sample rate {sample_rate}. Please resample to {TARGET_SAMPLE_RATE} Hz first."
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"Loading model: {args.model}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

    if args.device == "cuda":
        asr_model = asr_model.cuda()
    elif args.device == "auto":
        try:
            asr_model = asr_model.cuda()
            print("Using CUDA")
        except Exception:
            print("CUDA unavailable, using CPU")

    asr_model.eval()

    chunk_samples = int(args.chunk_seconds * sample_rate)
    stride_samples = int(args.stride_seconds * sample_rate)
    context_samples = int(args.context_seconds * sample_rate)

    if chunk_samples <= 0 or stride_samples <= 0 or context_samples <= 0:
        raise SystemExit("chunk/stride/context must be > 0")

    total_samples = len(audio)
    prev_hypothesis = ""
    stable_text = ""

    print("\nStreaming transcription:\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "chunk.wav"
        step = 0
        for end in range(chunk_samples, total_samples + stride_samples, stride_samples):
            step += 1
            end_idx = min(end, total_samples)
            start_idx = max(0, end_idx - context_samples)
            chunk = audio[start_idx:end_idx]
            sf.write(str(tmp_wav), chunk, sample_rate)

            prediction = asr_model.transcribe(
                audio=[str(tmp_wav)],
                batch_size=1,
                num_workers=0,
                use_lhotse=args.use_lhotse,
                return_hypotheses=False,
                verbose=False,
            )
            hypothesis = normalize_prediction(prediction).strip()

            common_prefix = longest_common_prefix_words(prev_hypothesis, hypothesis)
            if common_prefix and len(common_prefix) > len(stable_text):
                stable_text = common_prefix

            delta = hypothesis[len(stable_text):].strip() if hypothesis.startswith(stable_text) else hypothesis
            current_time = end_idx / sample_rate

            print(f"[{current_time:7.2f}s] stable:  {stable_text}")
            print(f"[{current_time:7.2f}s] partial: {delta}\n")

            prev_hypothesis = hypothesis

            if args.realtime:
                time.sleep(args.stride_seconds)

    final_text = prev_hypothesis.strip()
    print("Final transcript:\n")
    print(final_text)


if __name__ == "__main__":
    main()
