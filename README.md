# CPU SmolVLM-500M VQA Fine-Tune + GGUF Quantization Comparison

A portfolio-quality, CPU-only proof-of-pipeline for vision-language model fine-tuning. Trains a LoRA adapter on `HuggingFaceTB/SmolVLM-500M-Instruct` against a surgical VQA dataset, merges, converts to GGUF, quantizes with llama.cpp, and compares three runtime variants on a held-out test set.

## Clinical Disclaimer

> This model and pipeline are research artifacts. They are not validated for surgical guidance, diagnosis, treatment, or any patient-facing clinical use. Outputs must not influence patient care.

## Why these choices

- **Model: `HuggingFaceTB/SmolVLM-500M-Instruct`.** Apache-2.0, standard `transformers` (no `trust_remote_code`), natively supported by `llama.cpp convert_hf_to_gguf.py` for both the text decoder and the vision projector (`--mmproj`). A 500M-parameter VLM finishes a 250-step LoRA run on 4 CPU cores in roughly 30-90 minutes, not days. An earlier draft of this project targeted `vikhyatk/moondream2`; that model has no Moondream handler in `convert_hf_to_gguf.py` and is not listed in `tools/mtmd/README.md`, so a merged fine-tune cannot round-trip to GGUF through the documented path.
- **Dataset: `mmrech/pitvqa-sage-sft`.** Public, ungated, ~1 GB, surgical pituitary VQA in `messages` format with train/val/test splits. Cholec80-VQA, the original surgical target, requires manual access and was excluded.
- **Inference: `llama-server` (C binary from llama.cpp), not `llama-cpp-python`.** llama-cpp-python has no `SmolVLMChatHandler`; `llama-server` is the documented mtmd entrypoint.
- **CPU-only end-to-end.** Hard 16 GB / 4-core / 50 GB envelope.

## Hardware and runtime assumptions

- Ubuntu 22.04, 4 CPU cores, 16 GB RAM, 50 GB disk, no GPU.
- BF16 weights when `/proc/cpuinfo` reports `avx512_bf16`; FP32 fallback otherwise.
- Two-layer OOM defense: kernel cgroups v2 (`systemd-run --user --scope -p MemoryMax=13G`) plus an in-process `psutil` RSS guard at 12 GB.
- Realistic wall-clock budget (proof-scale, 500-train / 100-val / 200-test, 250 optimizer steps):
  - Data prep: ~2 min once the dataset is cached
  - LoRA training: ~30-90 min
  - Merge + GGUF conversion (base + finetuned): ~5-15 min
  - Quantization Q8_0 + Q4_K_M: ~2-5 min
  - Evaluation across three variants: ~5-15 min total
  - Total: under two hours on a typical 4-core x86 VPS.

## Repository layout

```
data/        # download + normalize PitVQA SAGE
train/       # custom CPU LoRA loop + merge
convert/     # llama.cpp clone/build, convert_hf_to_gguf.py wrapper, quantize
eval/        # pinned prompts, llama-server lifecycle, metrics, OpenAI-compat HTTP client
pipeline/    # preflight, pipeline_state, end-to-end resumable runner
common/      # shared utils: guards, atomic writes, hashing, status writer, seed
results/     # comparison.csv / .md (generated)
runs/        # per-run logs, status.json, checkpoints (generated)
tests/       # unit, AST static, slow smoke
```

## Setup (Mac, for development)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements.txt
```

On Mac, when building llama.cpp for development, prefer the portable flags so binaries also run on the VPS:

```bash
python convert/build_llama_cpp.py            # builds with -DGGML_NATIVE=OFF -DGGML_AVX2=ON
```

Never reuse Mac-built llama.cpp binaries on the VPS — rebuild them there.

## Setup (VPS, for the real run)

```bash
ssh <user>@<vps-ip>
tmux new -s smolvlm

# clone code
git clone <repo-url> ~/smolvlm-vqa-gguf
cd ~/smolvlm-vqa-gguf

# install
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements.txt

# run the full pipeline under a 13 GB cgroup memory cap
systemd-run --user --scope -p MemoryMax=13G -p MemorySwap=0 \
  python pipeline/run_all.py --resume 2>&1 | tee runs/latest.log
```

Detach with `Ctrl-b d`, reconnect with `tmux attach -t smolvlm`.

## Full pipeline command sequence

```bash
python data/prepare_dataset.py \
  --dataset mmrech/pitvqa-sage-sft \
  --train-samples 500 --val-samples 100 --test-samples 200
python pipeline/preflight.py
python train/finetune_lora.py --resume auto
python train/merge_lora.py \
  --adapter models/adapters/pitvqa-lora \
  --output  models/hf/smolvlm-500m-pitvqa-merged
python convert/build_llama_cpp.py
python convert/convert_to_gguf.py --model-dir HuggingFaceTB/SmolVLM-500M-Instruct --variant base
python convert/convert_to_gguf.py --model-dir models/hf/smolvlm-500m-pitvqa-merged --variant finetuned
python convert/quantize_gguf.py --input models/gguf/finetuned/text-f16.gguf --quant Q8_0
python convert/quantize_gguf.py --input models/gguf/finetuned/text-f16.gguf --quant Q4_K_M
python pipeline/run_comparison.py --threads 4
```

Or run everything resumable in one shot:

```bash
python pipeline/run_all.py --resume
```

## Variants compared

| variant | text GGUF | mmproj GGUF | weights size (approx) |
|---|---|---|---|
| `base-f16` | F16, base SmolVLM-500M | F16 | 1.0 GB |
| `finetuned-q8_0` | Q8_0, merged fine-tune | F16 | 0.55 GB |
| `finetuned-q4_k_m` | Q4_K_M, merged fine-tune | F16 | 0.30 GB |

(Stretch: add `base-q4_k_m` for a cleaner 2x2 comparison that isolates the quantization effect.)

## Outputs

- `results/comparison.csv` and `results/comparison.md` — accuracy/latency/RAM table
- `results/<variant>.json` — per-variant detail including per-`qa_type` accuracy and failure examples
- `runs/<run_id>/status.json` — live status, last error, recommended resume command
- `runs/<run_id>/train.log`, `runs/latest.log`
- `runs/pipeline_state.json` — config/script/dataset hashes guarding `.done` markers

## Monitoring on the VPS

```bash
tail -f runs/latest.log
cat  runs/run_all/status.json
free -h
df   -h
ps   aux --sort=-%mem | head
```

Optional Claude Code monitor prompt:

> Monitor this SmolVLM CPU fine-tuning run. Inspect `runs/latest.log` and `runs/*/status.json`. Do not edit code unless I ask. If the run fails, identify the root cause and give the exact resume command.

## How LoRA merge and GGUF conversion work here

1. PEFT injects rank-8 LoRA adapters into the SmolVLM text decoder's `q_proj`, `k_proj`, `v_proj`, `o_proj` linear layers only. Vision encoder and connector are explicitly frozen; an in-loader assertion fails the run if any trainable parameter falls outside the text decoder.
2. After training, `train/merge_lora.py` calls `PeftModel.merge_and_unload()` and `save_pretrained()` to write a standard HF checkpoint with the LoRA weights folded into the base linear layers.
3. `convert_hf_to_gguf.py` from llama.cpp reads the merged checkpoint twice: once to emit the text GGUF (F16), once with `--mmproj` to emit the vision projector GGUF (F16).
4. `llama-quantize` produces Q8_0 and Q4_K_M variants of the fine-tuned text GGUF. The mmproj remains F16 throughout.
5. `llama-server` loads `(text.gguf, mmproj.gguf)` and serves an OpenAI-compatible `/v1/chat/completions` endpoint that the evaluator hits over HTTP for each test sample. Templates are pinned (`--chat-template chatml`) to dodge llama.cpp issue [#21634](https://github.com/ggml-org/llama.cpp/issues/21634).

## Scaling path beyond proof scale

- Larger VLM: SmolVLM-2.2B-Instruct (same conversion path) or Qwen2.5-VL-3B/7B; expect proportional RAM/wall-clock growth.
- Real GPU node: drop the cgroups cap, set `DEVICE = "cuda"` in entrypoints (only after auditing every call site), rebuild llama.cpp with `-DGGML_CUDA=ON`, and increase `n_gpu_layers` on the server.
- Larger dataset subset: bump `--train-samples` to the full ~12 k PitVQA SAGE train split.
- Broader evaluation: add `Saint-lsy/EndoBench` as an external held-out benchmark.
- Faster conversion/quantization: parallelize Q8_0 and Q4_K_M when RAM allows; currently sequential by design on the 16 GB VPS.

## Testing

Fast tests (unit + AST static scan):

```bash
pytest tests/ --ignore=tests/test_training_smoke.py
```

Slow smoke (downloads the model, runs one optimizer step):

```bash
RUN_SLOW=1 pytest tests/test_training_smoke.py
```

The AST static scan in `tests/test_static_cpu_only.py` enforces:

- No `.cuda()` calls anywhere in the project.
- No `.to("cuda")` / `device_map="cuda"` / `n_gpu_layers>0` literal arguments.
- Every executable script under `data/`, `train/`, `convert/`, `pipeline/` declares `DEVICE = "cpu"`.
- `requirements.txt` does not pull any GPU-only or duplicate-inference dependency (`bitsandbytes`, `flash-attn`, `nvidia-*`, `llama-cpp-python`).

## Sources

- Model card: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
- Official SmolVLM fine-tuning notebook (LoRA target modules): https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb
- llama.cpp `convert_hf_to_gguf.py` (SmolVLM/Idefics3 support): https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- llama.cpp mtmd README: https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md
- llama.cpp issue #21634 (SmolVLM `--chat-template` workaround): https://github.com/ggml-org/llama.cpp/issues/21634
- PitVQA SAGE dataset: https://huggingface.co/datasets/mmrech/pitvqa-sage-sft
- Idefics3 modeling reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py
