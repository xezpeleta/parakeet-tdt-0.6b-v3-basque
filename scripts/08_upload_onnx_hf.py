#!/usr/bin/env python3
"""Upload both ONNX model packages to Hugging Face.

Usage:
    python scripts/08_upload_onnx_hf.py

Requires:
    pip install huggingface_hub
    HF_TOKEN env variable set (or logged in via `huggingface-cli login`)
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USER = "xezpeleta"

SHERPA_REPO_ID = f"{HF_USER}/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"
ONNX_ASR_REPO_ID = f"{HF_USER}/parakeet-tdt-0.6b-v3-basque-onnx-asr"

BASE_DIR = Path("/workspace/parakeet-basque")
SHERPA_MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-parakeet-tdt-0.6b-v3-basque"
ONNX_ASR_MODEL_DIR = BASE_DIR / "models" / "onnx-asr-parakeet-tdt-0.6b-v3-basque"
CARDS_DIR = BASE_DIR / "scripts" / "hf_cards"

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_repo(api: HfApi, repo_id: str) -> None:
    """Create the repo if it doesn't already exist."""
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            token=HF_TOKEN,
            exist_ok=True,
        )
        print(f"  ✓ Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Could not create repo {repo_id}: {e}")
        raise


def copy_readme(src: Path, dest_dir: Path) -> None:
    """Copy a model card README into the model directory as README.md."""
    dest = dest_dir / "README.md"
    shutil.copy2(src, dest)
    print(f"  ✓ README.md written to {dest}")


def upload_folder(api: HfApi, local_dir: Path, repo_id: str) -> None:
    """Upload all files in local_dir to the HF repo root."""
    print(f"  Uploading {local_dir} → {repo_id} …")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=HF_TOKEN,
        commit_message="Add INT8 ONNX model files and model card",
    )
    print(f"  ✓ Upload complete: https://huggingface.co/{repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api = HfApi()

    # ── 1) sherpa-onnx ────────────────────────────────────────────────────────
    print("\n=== sherpa-onnx ===")
    make_repo(api, SHERPA_REPO_ID)
    copy_readme(CARDS_DIR / "sherpa-onnx-README.md", SHERPA_MODEL_DIR)
    upload_folder(api, SHERPA_MODEL_DIR, SHERPA_REPO_ID)

    # ── 2) onnx-asr ───────────────────────────────────────────────────────────
    print("\n=== onnx-asr ===")
    make_repo(api, ONNX_ASR_REPO_ID)
    copy_readme(CARDS_DIR / "onnx-asr-README.md", ONNX_ASR_MODEL_DIR)
    upload_folder(api, ONNX_ASR_MODEL_DIR, ONNX_ASR_REPO_ID)

    print("\n=== Done ===")
    print(f"  sherpa-onnx : https://huggingface.co/{SHERPA_REPO_ID}")
    print(f"  onnx-asr    : https://huggingface.co/{ONNX_ASR_REPO_ID}")


if __name__ == "__main__":
    main()
