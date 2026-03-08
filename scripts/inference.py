#!/usr/bin/env python3
"""Run offline ASR inference with the Basque Parakeet model from Hugging Face.

Examples:
  python scripts/inference.py --audio /path/to/file.wav
  python scripts/inference.py --audio a.wav b.wav c.wav --batch-size 8
  python scripts/inference.py --audio-dir /path/to/wavs --pattern "*.wav" --output preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import nemo.collections.asr as nemo_asr

DEFAULT_MODEL = "itzune/parakeet-tdt-0.6b-v3-basque"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for Basque Parakeet model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model repo or local .nemo path")
    parser.add_argument("--audio", nargs="*", default=[], help="One or more audio files")
    parser.add_argument("--audio-dir", default="", help="Directory with audio files")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern used with --audio-dir")
    parser.add_argument("--batch-size", type=int, default=8, help="Transcription batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    parser.add_argument("--use-lhotse", action="store_true", help="Enable NeMo lhotse dataloader")
    parser.add_argument("--output", default="", help="Optional output JSONL path")
    return parser.parse_args()


def collect_audio_files(args: argparse.Namespace) -> List[str]:
    files: List[str] = [str(Path(audio).expanduser().resolve()) for audio in args.audio]

    if args.audio_dir:
        base_dir = Path(args.audio_dir).expanduser().resolve()
        files.extend(str(path.resolve()) for path in sorted(base_dir.glob(args.pattern)) if path.is_file())

    unique_files = list(dict.fromkeys(files))
    if not unique_files:
        raise SystemExit("No audio files provided. Use --audio and/or --audio-dir.")

    missing = [path for path in unique_files if not Path(path).exists()]
    if missing:
        raise SystemExit("Audio files not found:\n  - " + "\n  - ".join(missing))

    return unique_files


def to_text_list(predictions):
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    texts = []
    for item in predictions:
        texts.append(item.text if hasattr(item, "text") else str(item))
    return texts


def main() -> None:
    args = parse_args()
    audio_files = collect_audio_files(args)

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

    print(f"Transcribing {len(audio_files)} file(s)...")
    predictions = asr_model.transcribe(
        audio=audio_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_lhotse=args.use_lhotse,
        return_hypotheses=False,
        verbose=True,
    )
    texts = to_text_list(predictions)

    rows = []
    for path, text in zip(audio_files, texts):
        row = {"audio_filepath": path, "pred_text": text}
        rows.append(row)
        print(f"\n{path}\n{text}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved predictions: {out_path}")


if __name__ == "__main__":
    main()
