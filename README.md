# SmolVLM-500M VQA fine-tune + GGUF quantization

Proof-of-pipeline: fine-tune a 500M-param vision-language model on surgical VQA, quantize it two ways, compare. CPU-only on a 4-core / 16 GB VPS. ~3 hours end-to-end.

> Research artifact. Not for clinical use.

## Results

10-sample held-out test:

| Variant            | BERTScore F1 | Peak RAM | Latency / query |
|--------------------|--------------|----------|-----------------|
| `base-f16`         | 0.731        | 1520 MB  | 64.3 s          |
| `finetuned-q8_0`   | 0.723        | 1155 MB  | 61.8 s          |
| `finetuned-q4_k_m` | 0.703        | 1048 MB  | 63.3 s          |

Q4_K_M cuts RAM 31% for a 3.8% F1 cost. Latency is image-token bound, not weight-precision bound.

![comparison chart](results/comparison.png)

## Pipeline

1. Prep PitVQA SAGE (500 train / 100 val / 10 test)
2. LoRA fine-tune SmolVLM-500M (r=8, 125 steps, FP32 on CPU)
3. Merge LoRA into base
4. GGUF convert: base + merged (text + mmproj)
5. Quantize merged → Q8_0 and Q4_K_M
6. Eval all 3 variants via `llama-server`

Every stage writes a `.done` marker. Restart auto-skips finished stages. Config / script / dataset hashes invalidate stale markers when anything upstream changes.

## Run

```bash
git clone https://github.com/shhubbh/moondream-finetune.git
cd moondream-finetune

python3.10 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements.txt
apt-get install -y build-essential cmake

python pipeline/run_all.py --resume 2>&1 | tee runs/latest.log
```

## Layout

```
data/      dataset prep
train/     LoRA + merge
convert/   llama.cpp build + GGUF + quantize
eval/      llama-server + metrics + plot
pipeline/  preflight + orchestrator
common/    guards, hashing, status
results/   comparison.csv / .md / .png
```

## Stack

SmolVLM-500M-Instruct (Apache 2.0) · PitVQA SAGE (public, ungated) · `transformers` + `peft` · `llama.cpp` · `llama-server`
