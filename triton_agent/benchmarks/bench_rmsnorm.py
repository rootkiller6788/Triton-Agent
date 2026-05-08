"""Benchmark runner for RMSNorm across multiple shapes."""

import json
import time
from pathlib import Path


SHAPES = [
    {"B": 1, "T": 1, "D": 64},
    {"B": 1, "T": 128, "D": 2048},
    {"B": 8, "T": 2048, "D": 4096},
    {"B": 32, "T": 2048, "D": 4096},
    {"B": 64, "T": 128, "D": 4096},
]

DTYPES = ["fp16", "bf16"]


def run():
    """Run the benchmark suite and print a summary table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold]RMSNorm Benchmark Suite[/bold]")

    table = Table(title="RMSNorm Multi-Shape Results")
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Latency(p50)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status")

    for shape in SHAPES:
        for dtype in DTYPES:
            shape_str = f"B={shape['B']},T={shape['T']},D={shape['D']}"
            console.print(f"  Benchmarking {shape_str} {dtype}...")

            table.add_row(shape_str, dtype, "—", "—", "pending")

    console.print(table)
    console.print("Run with GPU to populate real values.")


if __name__ == "__main__":
    run()
