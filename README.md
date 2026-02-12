# Cognitive Cost of LLM Quantization

**Thesis:** Quantization trades reasoning quality for efficiency — but how much? This project measures that tradeoff using CARS (Cognitive Accuracy per Resource-Second), a unified metric that captures accuracy, memory, and speed in a single number.

## The CARS Metric

```
CARS = Reasoning Accuracy / (VRAM_GB × Latency_s)
```

| Component | What it measures |
|-----------|-----------------|
| **Reasoning Accuracy** | % correct on reasoning benchmarks (GSM8K, ARC) |
| **VRAM_GB** | Peak GPU memory during inference |
| **Latency_s** | Average per-sample inference time |

Higher CARS = better reasoning per unit of compute resource.

## Models Under Test

| Model | Parameters | Quantization | Expected VRAM |
|-------|-----------|--------------|---------------|
| Llama-3.2-3B | 3B | FP16 (baseline) | ~6 GB |
| Llama-3.2-3B | 3B | AWQ 4-bit | ~2 GB |
| Llama-3.2-3B | 3B | GPTQ 4-bit | ~2 GB |
| Llama-3-8B | 8B | 8-bit bitsandbytes (baseline) | ~9 GB |
| Llama-3-8B | 8B | AWQ 4-bit | ~5 GB |
| Llama-3-8B | 8B | GPTQ 4-bit | ~5 GB |

## Benchmarks

- **GSM8K** — Grade school math word problems (multi-step reasoning)
- **ARC-Easy** — Elementary science questions
- **ARC-Challenge** — Harder science questions requiring reasoning

## Quick Start

### Install

```bash
pip install -e .
```

### Run a benchmark

```bash
# Small model, quick validation
python -m src.benchmark --model meta-llama/Llama-3.2-3B --tasks arc_easy --limit 10

# Full benchmark suite
python -m src.benchmark --model meta-llama/Llama-3.2-3B --tasks gsm8k arc_easy arc_challenge
```

### View CARS scores

```bash
python -m src.cars results/
```

## Colab Setup

This project runs on Google Colab Free (T4 GPU). See `notebooks/colab_setup.ipynb` for SSH tunnel setup, then run:

```bash
./setup_colab.sh
```

## Project Structure

```
src/
├── cars.py          # CARS metric calculator + comparison table
├── benchmark.py     # Benchmark runner (wraps lm-evaluation-harness)
└── models.py        # Model loading helpers (FP16/AWQ/GPTQ)
notebooks/
└── colab_setup.ipynb  # Colab SSH tunnel setup
results/             # Benchmark output JSONs
setup_colab.sh       # One-command Colab provisioning
```

## License

MIT
