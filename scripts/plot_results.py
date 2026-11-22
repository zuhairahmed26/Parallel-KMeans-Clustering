#!/usr/bin/env python3
"""Create Phase 6 benchmark plots from benchmark.py's summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_successful(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    successful = [r for r in rows if r.get("status") == "OK"]
    if not successful:
        raise ValueError("summary CSV contains no fully successful benchmark configurations")
    return successful


def save_runtime_comparison(rows: list[dict], out: Path) -> None:
    strong = [r for r in rows if r["experiment"] == "strong"]
    labels, values, colors = [], [], []
    palette = {"serial": "#555555", "mpi-only": "#2878B5", "hybrid": "#E87500"}
    for row in sorted(strong, key=lambda r: (int(r["total_workers"]), int(r["mpi_ranks"]), int(r["threads_per_rank"]))):
        if not row["mean_runtime_seconds"]:
            continue
        if row["mode"] == "serial":
            labels.append("Serial\n1 worker")
        else:
            labels.append(f"{row['mode']}\n{row['mpi_ranks']}×{row['threads_per_rank']}\n{row['total_workers']} workers")
        values.append(float(row["mean_runtime_seconds"]))
        colors.append(palette[row["mode"]])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Mean algorithm runtime (seconds)")
    ax.set_title("Strong-scaling runtime comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=[Patch(color=color, label=mode) for mode, color in palette.items()])
    fig.tight_layout()
    fig.savefig(out / "runtime_comparison.png", dpi=180)
    plt.close(fig)


def strong_lines(rows: list[dict], field: str, ylabel: str, title: str, filename: str, out: Path) -> None:
    strong = [r for r in rows if r["experiment"] == "strong" and r["mode"] != "serial"]
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {1: "o", 2: "s", 4: "^"}
    for ranks in sorted({int(r["mpi_ranks"]) for r in strong}):
        series = sorted((r for r in strong if int(r["mpi_ranks"]) == ranks),
                        key=lambda r: int(r["threads_per_rank"]))
        points = [(int(r["total_workers"]), float(r[field])) for r in series if r[field] not in ("", None)]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], marker=markers.get(ranks, "o"),
                    linewidth=1.8, label=f"{ranks} MPI rank(s)")
    ax.set_xlabel("Total workers (MPI ranks × OpenMP threads per rank)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Headroom keeps the legend and highest point clear in the efficiency plot,
    # where measured efficiency can exceed 1 because of short-run noise.
    ax.margins(y=0.20)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)


def weak_plot(rows: list[dict], out: Path) -> None:
    weak = [r for r in rows if r["experiment"] == "weak" and r["mode"] != "serial"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode, marker, label in [("mpi-only", "o", "MPI-only (1 thread/rank)"),
                                ("hybrid", "s", "Hybrid MPI + OpenMP")]:
        series = sorted((r for r in weak if r["mode"] == mode), key=lambda r: int(r["total_workers"]))
        if series:
            ax.plot([int(r["total_workers"]) for r in series],
                    [float(r["mean_runtime_seconds"]) for r in series],
                    marker=marker, linewidth=1.8, label=label)
    ax.set_xlabel("Total workers (MPI ranks × OpenMP threads per rank)")
    ax.set_ylabel("Mean algorithm runtime (seconds)")
    ax.set_title("Weak scaling: workload grows with worker count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "weak_scaling_runtime.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    rows = read_successful(args.summary_csv)
    output = args.output_dir or args.summary_csv.parent
    output.mkdir(parents=True, exist_ok=True)
    save_runtime_comparison(rows, output)
    strong_lines(rows, "mean_runtime_seconds", "Mean runtime (seconds)",
                 "Strong scaling: runtime vs total workers", "strong_scaling_runtime.png", output)
    strong_lines(rows, "mean_speedup", "Speedup (serial runtime / parallel runtime)",
                 "Strong scaling speedup", "speedup.png", output)
    strong_lines(rows, "parallel_efficiency", "Efficiency (speedup / total workers)",
                 "Strong scaling parallel efficiency", "efficiency.png", output)
    if any(r["experiment"] == "weak" for r in rows):
        weak_plot(rows, output)
    print(f"Plots saved in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
