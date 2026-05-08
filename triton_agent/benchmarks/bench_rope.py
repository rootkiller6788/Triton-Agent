"""Benchmark runner for RoPE across multiple shapes."""

from pathlib import Path


SHAPES = [
    {"B": 1, "H": 32, "T": 2048, "D": 128},
    {"B": 4, "H": 32, "T": 2048, "D": 128},
    {"B": 8, "H": 32, "T": 4096, "D": 128},
    {"B": 32, "H": 32, "T": 1024, "D": 128},
]

DTYPES = ["fp16", "bf16"]


def run():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold]RoPE Benchmark Suite[/bold]")

    table = Table(title="RoPE Multi-Shape Results")
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Latency(p50)", justify="right")
    table.add_column("Speedup", justify="right")

    for shape in SHAPES:
        for dtype in DTYPES:
            shape_str = f"B={shape['B']},H={shape['H']},T={shape['T']},D={shape['D']}"
            table.add_row(shape_str, dtype, "—", "—")

    console.print(table)
    console.print("Run with GPU for real values.")


if __name__ == "__main__":
    run()
