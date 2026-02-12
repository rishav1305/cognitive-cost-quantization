"""CARS (Cognitive Accuracy per Resource-Second) metric calculator.

CARS = Reasoning Accuracy / (VRAM_GB × Latency_s)

Higher CARS means better reasoning per unit of compute resource.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tabulate import tabulate


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    model: str
    task: str
    quantization: str  # "fp16", "8bit", "awq-4bit", "gptq-4bit"
    accuracy: float  # 0.0 - 1.0
    vram_gb: float
    latency_s: float  # average per-sample
    cars_score: float = field(init=False)
    num_samples: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cars_score = compute_cars(self.accuracy, self.vram_gb, self.latency_s)


def compute_cars(accuracy: float, vram_gb: float, latency_s: float) -> float:
    """Compute CARS = Reasoning Accuracy / (VRAM_GB × Latency_s).

    Args:
        accuracy: Reasoning accuracy as a fraction (0.0 to 1.0).
        vram_gb: Peak GPU memory usage in gigabytes.
        latency_s: Average per-sample inference latency in seconds.

    Returns:
        CARS score. Higher is better.
    """
    if vram_gb <= 0 or latency_s <= 0:
        return 0.0
    return accuracy / (vram_gb * latency_s)


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load benchmark results from a directory of JSON files."""
    results_dir = Path(path)
    results: list[BenchmarkResult] = []

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                results.append(_parse_result(entry))
        else:
            results.append(_parse_result(data))

    return results


def _parse_result(data: dict) -> BenchmarkResult:
    """Parse a dict into a BenchmarkResult."""
    return BenchmarkResult(
        model=data["model"],
        task=data["task"],
        quantization=data["quantization"],
        accuracy=data["accuracy"],
        vram_gb=data["vram_gb"],
        latency_s=data["latency_s"],
        num_samples=data.get("num_samples", 0),
        metadata=data.get("metadata", {}),
    )


def save_result(result: BenchmarkResult, path: str | Path) -> Path:
    """Save a single benchmark result as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return out_path


def compare_models(results: list[BenchmarkResult]) -> str:
    """Format a comparison table of benchmark results."""
    if not results:
        return "No results to compare."

    headers = ["Model", "Quant", "Task", "Accuracy", "VRAM (GB)", "Latency (s)", "CARS"]
    rows = []
    for r in sorted(results, key=lambda x: x.cars_score, reverse=True):
        rows.append([
            r.model.split("/")[-1],
            r.quantization,
            r.task,
            f"{r.accuracy:.4f}",
            f"{r.vram_gb:.2f}",
            f"{r.latency_s:.3f}",
            f"{r.cars_score:.4f}",
        ])

    return tabulate(rows, headers=headers, tablefmt="github")


def main() -> None:
    """CLI entry point: python -m src.cars results/"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.cars <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    if not results:
        print(f"No result files found in {results_dir}/")
        sys.exit(1)

    print(f"\n=== CARS Leaderboard ({len(results)} results) ===\n")
    print(compare_models(results))
    print()


if __name__ == "__main__":
    main()
