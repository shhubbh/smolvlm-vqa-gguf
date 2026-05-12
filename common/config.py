"""Project-wide constants and defaults. Importing this module asserts CPU-only mode."""

from __future__ import annotations

import os

DEVICE = "cpu"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
MODEL_REVISION = "main"

DATASET_ID = "mmrech/pitvqa-sage-sft"

DEFAULT_TRAIN_SAMPLES = 500
DEFAULT_VAL_SAMPLES = 100
DEFAULT_TEST_SAMPLES = 200
SEED = 42

NUM_THREADS = 4

MAX_OPTIMIZER_STEPS = 250
GRAD_ACCUM_STEPS = 8
BATCH_SIZE = 1
LR = 1e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

CHECKPOINT_EVERY_STEPS = 25
CHECKPOINT_EVERY_SECONDS = 60 * 60

MEMORY_GUARD_GB = 12.0
DISK_GUARD_FREE_GB = 3.0
PREFLIGHT_FREE_RAM_GB = 4.0
PREFLIGHT_FREE_DISK_GB = 10.0

# Stretch: add a 4th variant (base-q4_k_m) for a clean 2x2 grid.
VARIANTS = ("base-f16", "finetuned-q8_0", "finetuned-q4_k_m")

EVAL_TEMPERATURE = 0.0
EVAL_SEED = 42
EVAL_MAX_TOKENS = 64
EVAL_CHAT_TEMPLATE = "chatml"

BERTSCORE_MODEL = "distilbert-base-uncased"
BERTSCORE_NUM_LAYERS = 5

LLAMA_CPP_TAG = "b6912"

PATHS = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "data_images": "data/processed/images",
    "train_manifest": "data/processed/train.jsonl",
    "val_manifest": "data/processed/val.jsonl",
    "test_manifest": "data/processed/test.jsonl",
    "adapter": "models/adapters/pitvqa-lora",
    "merged_hf": "models/hf/smolvlm-500m-pitvqa-merged",
    "gguf_base": "models/gguf/base",
    "gguf_finetuned": "models/gguf/finetuned",
    "llama_cpp_dir": "convert/llama.cpp",
    "llama_cpp_bin": "convert/llama.cpp/build/bin",
    "runs": "runs",
    "results_csv": "results/comparison.csv",
    "results_md": "results/comparison.md",
}
