"""Benchmark runner for Fused Bias GELU across multiple shapes."""

from pathlib import Path


SHAPES = [
    {"B": 1, "T": 1, "D": 64},
    {"B": 1, "T": 128, "D": 2048},
    {"B": 8, "T": 2048, "D": 4096},
    {"B": 32, "T": 2048, "D": 4096},
    {"B": 64, "T": 256, "D": 4096},
]

DTYPES = ["fp16", "bf16"]


def run():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold]Fused Bias GELU Benchmark Suite[/bold]")

    table = Table(title="Fused Bias GELU Multi-Shape Results")
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Latency(p50)", justify="right")
    table.add_column("Speedup", justify="right")

    for shape in SHAPES:
        for dtype in DTYPES:
            shape_str = f"B={shape['B']},T={shape['T']},D={shape['D']}"
            table.add_row(shape_str, dtype, "—", "—")

    console.print(table)
    console.print("Run with GPU for real values.")


if __name__ == "__main__":
    run()
