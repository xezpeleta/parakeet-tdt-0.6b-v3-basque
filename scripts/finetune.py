#!/usr/bin/env python3
"""
Fine-tune parakeet-tdt-0.6b-v3 on Basque data.
Run inside the NeMo Docker container.
"""
import os
import sys
import json
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DATA_DIR = "/workspace/parakeet-basque/data"
RESULTS_DIR = "/workspace/parakeet-basque/results"
EXP_NAME = "parakeet-tdt-basque"

# Training hyperparameters
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
MAX_STEPS = 50000
VAL_CHECK_INTERVAL = 1000
BATCH_SIZE = 8           # Conservative for L40 48GB; increase if no OOM
ACCUMULATE_GRAD = 8      # Effective batch size = 8 * 8 = 64
NUM_WORKERS = 4
PRECISION = "bf16-mixed"

# Check for tarred dataset
TRAIN_TARRED_DIR = os.path.join(DATA_DIR, "train_tarred")
USE_TARRED = os.path.isdir(TRAIN_TARRED_DIR) and os.path.exists(
    os.path.join(TRAIN_TARRED_DIR, "tarred_audio_manifest.json")
)

# Manifest paths
if USE_TARRED:
    TRAIN_MANIFEST = os.path.join(TRAIN_TARRED_DIR, "tarred_audio_manifest.json")
    # Detect number of tar shards
    tar_files = sorted([f for f in os.listdir(TRAIN_TARRED_DIR) if f.endswith(".tar")])
    num_shards = len(tar_files)
    TRAIN_TARRED_PATHS = os.path.join(
        TRAIN_TARRED_DIR,
        "audio__OP_0..{}_CL_.tar".format(num_shards - 1)
    )
    print("Using tarred dataset: {} shards".format(num_shards))
else:
    TRAIN_MANIFEST = os.path.join(DATA_DIR, "train_manifest.json")
    print("Using regular (non-tarred) dataset")

DEV_MANIFEST = os.path.join(DATA_DIR, "dev_manifest.json")

# Verify manifests exist
for path in [TRAIN_MANIFEST, DEV_MANIFEST]:
    if not os.path.exists(path):
        alt = path.replace("train_tarred/", "")
        if os.path.exists(alt):
            print("Using alternative: {}".format(alt))
        else:
            print("ERROR: Manifest not found: {}".format(path))
            sys.exit(1)

# ============================================================================
# Load pretrained model
# ============================================================================
print("=" * 60)
print("Loading pretrained model: {}".format(MODEL_NAME))
print("=" * 60)

try:
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
except Exception as e:
    print("Error with ASRModel.from_pretrained: {}".format(e))
    print("Trying EncDecHybridRNNTCTCBPEModel...")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=MODEL_NAME)

print("Model type: {}".format(type(asr_model).__name__))
print("Model parameters: {:.1f}M".format(sum(p.numel() for p in asr_model.parameters()) / 1e6))

# ============================================================================
# Update model config for fine-tuning
# ============================================================================
print("\nUpdating model configuration for fine-tuning...")

cfg = asr_model.cfg

with open_dict(cfg):
    # ---- Training dataset ----
    # Disable Lhotse dataloader (NeMo 2.7 default) — use traditional NeMo dataloader
    cfg.train_ds.use_lhotse = False
    if USE_TARRED:
        cfg.train_ds.is_tarred = True
        cfg.train_ds.tarred_audio_filepaths = TRAIN_TARRED_PATHS
        cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
    else:
        cfg.train_ds.is_tarred = False
        cfg.train_ds.manifest_filepath = TRAIN_MANIFEST

    cfg.train_ds.batch_size = BATCH_SIZE
    cfg.train_ds.num_workers = NUM_WORKERS
    cfg.train_ds.pin_memory = True
    cfg.train_ds.shuffle = True
    cfg.train_ds.min_duration = 0.5
    cfg.train_ds.max_duration = 20.0
    # Set text field properly (default config uses 'answer' for Canary models)
    cfg.train_ds.text_field = "text"

    # ---- Validation dataset ----
    cfg.validation_ds.use_lhotse = False
    cfg.validation_ds.manifest_filepath = DEV_MANIFEST
    cfg.validation_ds.batch_size = BATCH_SIZE
    cfg.validation_ds.num_workers = NUM_WORKERS
    cfg.validation_ds.text_field = "text"

    # ---- Optimizer ----
    cfg.optim.name = "adamw"
    cfg.optim.lr = LEARNING_RATE
    cfg.optim.weight_decay = 1e-3

    # ---- Scheduler ----
    cfg.optim.sched.name = "CosineAnnealing"
    cfg.optim.sched.warmup_steps = WARMUP_STEPS
    cfg.optim.sched.min_lr = 1e-6

    # ---- Data augmentation (SpecAugment — already in default config, ensure enabled) ----
    if hasattr(cfg, "spec_augment") and cfg.spec_augment is not None:
        cfg.spec_augment.freq_masks = 2
        cfg.spec_augment.freq_width = 27
        cfg.spec_augment.time_masks = 10
        cfg.spec_augment.time_width = 0.05

# Apply updated config
asr_model.setup_training_data(cfg.train_ds)
asr_model.setup_validation_data(cfg.validation_ds)
asr_model.setup_optimization(cfg.optim)

# ============================================================================
# Setup Trainer
# ============================================================================
print("\nSetting up trainer...")

# W&B logger configuration
wandb_project = os.environ.get("WANDB_PROJECT", "parakeet-tdt-basque")
wandb_entity = os.environ.get("WANDB_ENTITY", None)
wandb_api_key = os.environ.get("WANDB_API_KEY", None)

use_wandb = wandb_api_key is not None and wandb_api_key != ""

if use_wandb:
    print("W&B logging enabled: project={}".format(wandb_project))
else:
    print("W&B logging disabled (no WANDB_API_KEY). Using TensorBoard only.")

# Trainer configuration
trainer_config = {
    "devices": 1,
    "accelerator": "gpu",
    "max_steps": MAX_STEPS,
    "val_check_interval": VAL_CHECK_INTERVAL,
    "accumulate_grad_batches": ACCUMULATE_GRAD,
    "precision": PRECISION,
    "log_every_n_steps": 50,
    "enable_checkpointing": False,  # Managed by exp_manager
    "logger": False,                # Managed by exp_manager
    "num_sanity_val_steps": 2,
}

trainer = pl.Trainer(**trainer_config)

# ============================================================================
# Setup Experiment Manager (handles checkpoints, logging, W&B)
# ============================================================================
exp_manager_config = {
    "exp_dir": RESULTS_DIR,
    "name": EXP_NAME,
    "resume_if_exists": True,
    "resume_ignore_no_checkpoint": True,
    "create_tensorboard_logger": True,
    "create_checkpoint_callback": True,
    "checkpoint_callback_params": {
        "monitor": "val_wer",
        "mode": "min",
        "save_top_k": 3,
        "always_save_nemo": True,
    },
}

if use_wandb:
    exp_manager_config["create_wandb_logger"] = True
    exp_manager_config["wandb_logger_kwargs"] = {
        "project": wandb_project,
        "name": EXP_NAME,
    }
    if wandb_entity:
        exp_manager_config["wandb_logger_kwargs"]["entity"] = wandb_entity

exp_manager_cfg = OmegaConf.create(exp_manager_config)
exp_manager(trainer, cfg=exp_manager_cfg)

# ============================================================================
# Start Fine-tuning
# ============================================================================
print("\n" + "=" * 60)
print("Starting fine-tuning!")
print("  Model: {}".format(MODEL_NAME))
print("  Train manifest: {}".format(TRAIN_MANIFEST))
print("  Dev manifest: {}".format(DEV_MANIFEST))
print("  Tarred: {}".format(USE_TARRED))
print("  Batch size: {} x {} = {} effective".format(BATCH_SIZE, ACCUMULATE_GRAD, BATCH_SIZE * ACCUMULATE_GRAD))
print("  Learning rate: {}".format(LEARNING_RATE))
print("  Max steps: {}".format(MAX_STEPS))
print("  Precision: {}".format(PRECISION))
print("  W&B: {}".format("enabled" if use_wandb else "disabled"))
print("=" * 60)
print()

trainer.fit(asr_model)

print("\n" + "=" * 60)
print("Fine-tuning complete!")
print("Results directory: {}/{}".format(RESULTS_DIR, EXP_NAME))
print("=" * 60)
