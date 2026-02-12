"""Benchmark runner for measuring reasoning accuracy, VRAM, and latency.

Wraps EleutherAI's lm-evaluation-harness for standardized benchmarking,
then computes CARS scores for each model/task combination.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from src.cars import BenchmarkResult, compare_models, save_result
from src.models import get_vram_usage, load_model, reset_vram_tracking

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["gsm8k", "arc_easy", "arc_challenge"]
DEFAULT_RESULTS_DIR = "results"


def run_benchmark(
    model_id: str,
    tasks: list[str],
    quantization: str | None = None,
    bits: int | None = None,
    limit: int | None = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
    batch_size: int = 1,
) -> list[BenchmarkResult]:
    """Run benchmarks on a model and save results.

    Args:
        model_id: HuggingFace model ID.
        tasks: List of benchmark tasks to run.
        quantization: Quantization method (None, "8bit", "awq", "gptq").
        bits: Bit width for quantization.
        limit: Max samples per task (None for full dataset).
        results_dir: Directory to save result JSONs.
        batch_size: Batch size for evaluation.

    Returns:
        List of BenchmarkResult objects.
    """
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown task: {task}. Supported: {SUPPORTED_TASKS}")

    reset_vram_tracking()
    model, tokenizer, model_config = load_model(
        model_id, quantization=quantization, bits=bits
    )

    results: list[BenchmarkResult] = []

    for task in tasks:
        logger.info("Running %s on %s (%s)...", task, model_id, model_config.label)
        result = _run_single_task(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            task=task,
            limit=limit,
            batch_size=batch_size,
        )
        results.append(result)

        # Save immediately after each task
        model_short = model_id.split("/")[-1]
        filename = f"{model_short}_{model_config.label}_{task}.json"
        out_path = save_result(result, Path(results_dir) / filename)
        logger.info("Saved %s (CARS=%.4f)", out_path, result.cars_score)

    return results


def _run_single_task(
    model: torch.nn.Module,
    tokenizer,
    model_config,
    task: str,
    limit: int | None,
    batch_size: int,
) -> BenchmarkResult:
    """Run a single evaluation task using lm-evaluation-harness."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    reset_vram_tracking()

    # Wrap model for lm-eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    # Run evaluation
    start_time = time.time()
    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        limit=limit,
        batch_size=batch_size,
    )
    total_time = time.time() - start_time

    # Extract accuracy from results
    task_results = eval_results["results"][task]
    accuracy = _extract_accuracy(task_results, task)

    # Calculate metrics
    peak_vram = get_vram_usage()
    num_samples = limit if limit else _get_task_size(task_results)
    per_sample_latency = total_time / max(num_samples, 1)

    return BenchmarkResult(
        model=model_config.model_id,
        task=task,
        quantization=model_config.label,
        accuracy=accuracy,
        vram_gb=peak_vram,
        latency_s=per_sample_latency,
        num_samples=num_samples,
        metadata={
            "total_time_s": total_time,
            "batch_size": batch_size,
            "raw_results": {
                k: v
                for k, v in task_results.items()
                if isinstance(v, (int, float))
            },
        },
    )


def _extract_accuracy(task_results: dict, task: str) -> float:
    """Extract the primary accuracy metric from lm-eval results."""
    # lm-eval uses different metric names per task
    metric_keys = [
        "acc,none",
        "acc_norm,none",
        "exact_match,strict-match",
        "exact_match,none",
    ]
    for key in metric_keys:
        if key in task_results:
            return float(task_results[key])

    # Fallback: look for any accuracy-like metric
    for key, value in task_results.items():
        if "acc" in key and isinstance(value, (int, float)):
            return float(value)

    logger.warning("No accuracy metric found for %s, returning 0.0", task)
    return 0.0


def _get_task_size(task_results: dict) -> int:
    """Try to get the number of samples evaluated."""
    for key in ["num_samples", "num_examples"]:
        if key in task_results:
            return int(task_results[key])
    return 0


def main() -> None:
    """CLI entry point: python -m src.benchmark --model <id> --tasks <tasks>"""
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks with CARS scoring"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=SUPPORTED_TASKS,
        help=f"Benchmark tasks (default: {SUPPORTED_TASKS})",
    )
    parser.add_argument(
        "--quantization",
        choices=["8bit", "awq", "gptq"],
        default=None,
        help="Quantization method",
    )
    parser.add_argument(
        "--bits", type=int, default=None, help="Bit width for quantization"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples per task"
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR, help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    results = run_benchmark(
        model_id=args.model,
        tasks=args.tasks,
        quantization=args.quantization,
        bits=args.bits,
        limit=args.limit,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
    )

    print(f"\n=== Benchmark Complete ({len(results)} tasks) ===\n")
    print(compare_models(results))

    for r in results:
        print(f"\n{r.model} ({r.quantization}) on {r.task}:")
        print(f"  Accuracy: {r.accuracy:.4f}")
        print(f"  VRAM:     {r.vram_gb:.2f} GB")
        print(f"  Latency:  {r.latency_s:.3f} s/sample")
        print(f"  CARS:     {r.cars_score:.4f}")


if __name__ == "__main__":
    main()
