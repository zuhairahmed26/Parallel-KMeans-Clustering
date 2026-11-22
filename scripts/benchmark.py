#!/usr/bin/env python3
"""Reproducible serial/MPI/OpenMP K-Means benchmark runner.

Uses only Python's standard library. Runtime is read from the applications so
it measures comparable algorithm sections, not Python/MPI process startup.
Every failed run is retained in raw_results.csv and causes a non-zero exit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


STRONG_CONFIGS = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4),
                  (4, 1), (4, 2), (4, 4)]
WEAK_CONFIGS = [(1, 1), (2, 1), (2, 2), (4, 2)]
OUTPUT_RE = {
    "runtime": re.compile(r"Runtime \(seconds\):\s*([0-9.eE+-]+)"),
    "iterations": re.compile(r"Iterations:\s*(\d+)"),
    "inertia": re.compile(r"Inertia \(SSE\):\s*([0-9.eE+-]+)"),
    "threads": re.compile(r"OpenMP threads per rank:\s*(\d+)"),
}
CENTROID_RE = re.compile(r"^\s*\d+:\s*\[(.*)\]\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_build = Path.home() / "parallel-kmeans-build"
    parser.add_argument("--serial-exe", default=str(default_build / "kmeans"), help="serial executable path")
    parser.add_argument("--mpi-exe", default=str(default_build / "kmeans_mpi"), help="MPI/OpenMP executable path")
    parser.add_argument("--mpirun", default="mpirun", help="MPI launcher")
    parser.add_argument("--output-root", default="results", help="parent folder for timestamped output")
    parser.add_argument("--strong-points", type=int, default=100000)
    parser.add_argument("--weak-base-points", type=int, default=20000)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--centroid-tolerance", type=float, default=1e-5,
                        help="absolute maximum allowed centroid difference")
    parser.add_argument("--sse-relative-tolerance", type=float, default=1e-6,
                        help="relative maximum allowed SSE difference")
    parser.add_argument("--skip-weak", action="store_true")
    args = parser.parse_args()
    positive = ("strong_points", "weak_base_points", "dimensions", "clusters",
                "max_iterations", "repetitions")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.clusters > args.strong_points or args.clusters > args.weak_base_points:
        parser.error("clusters must not exceed either configured point count")
    if args.tolerance < 0 or not math.isfinite(args.tolerance):
        parser.error("--tolerance must be finite and non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def parse_model(output: str, serial: bool) -> dict:
    model = {}
    for key, pattern in OUTPUT_RE.items():
        match = pattern.search(output)
        if match:
            model[key] = float(match.group(1)) if key in ("runtime", "inertia") else int(match.group(1))
    if serial:
        model["threads"] = 1
    centers = []
    for line in output.splitlines():
        match = CENTROID_RE.match(line)
        if match:
            centers.extend(float(value.strip()) for value in match.group(1).split(","))
    model["centroids"] = centers
    required = ("runtime", "iterations", "inertia")
    missing = [key for key in required if key not in model]
    if missing or not centers:
        raise ValueError(f"could not parse program output fields: {', '.join(missing or ['centroids'])}")
    return model


def algorithm_args(args: argparse.Namespace, points: int) -> list[str]:
    return ["--points", str(points), "--dimensions", str(args.dimensions),
            "--clusters", str(args.clusters), "--max-iterations", str(args.max_iterations),
            "--tolerance", str(args.tolerance), "--seed", str(args.seed)]


def command_for(args: argparse.Namespace, mode: str, ranks: int, threads: int,
                points: int) -> list[str]:
    common = algorithm_args(args, points)
    if mode == "serial":
        return [args.serial_exe, *common]
    return [args.mpirun, "--oversubscribe", "-n", str(ranks), args.mpi_exe,
            *common, "--threads", str(threads)]


def run_one(args: argparse.Namespace, experiment: str, mode: str, ranks: int,
            threads: int, points: int, repetition: int, reference: dict | None) -> tuple[dict, dict | None]:
    command = command_for(args, mode, ranks, threads, points)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    row = {
        "experiment": experiment,
        "repetition": repetition,
        "mode": mode,
        "mpi_ranks": ranks,
        "threads_per_rank": threads,
        "total_workers": ranks * threads,
        "points": points,
        "dimensions": args.dimensions,
        "clusters": args.clusters,
        "seed": args.seed,
        "max_iterations_config": args.max_iterations,
        "tolerance_config": args.tolerance,
        "runtime_seconds": "",
        "iterations": "",
        "inertia_sse": "",
        "observed_threads": "",
        "centroid_max_abs_diff": "",
        "sse_abs_diff": "",
        "correctness_pass": "not_checked",
        "reproducibility_pass": "not_checked",
        "serial_reference_seconds": "",
        "speedup": "",
        "efficiency": "",
        "exit_code": "",
        "status": "failed",
        "command": shlex.join(command),
        "diagnostic": "",
    }
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env,
                                   timeout=args.timeout_seconds, check=False)
        row["exit_code"] = completed.returncode
        output = completed.stdout
        if completed.returncode != 0:
            row["diagnostic"] = output[-4000:]
            return row, None
        model = parse_model(output, mode == "serial")
        row["runtime_seconds"] = model["runtime"]
        row["iterations"] = model["iterations"]
        row["inertia_sse"] = model["inertia"]
        row["observed_threads"] = model["threads"]
        if model["threads"] != threads:
            row["status"] = "incorrect"
            row["diagnostic"] = f"requested {threads} threads but observed {model['threads']}"
            return row, model
        if mode != "serial" and reference is None:
            row["status"] = "unvalidated"
            row["correctness_pass"] = "false"
            row["diagnostic"] = "serial reference failed or was unavailable for this dataset"
            return row, model
        row["status"] = "ok"
        if reference is not None:
            centroid_diff = (max(abs(a - b) for a, b in zip(model["centroids"], reference["centroids"]))
                             if len(model["centroids"]) == len(reference["centroids"]) else math.inf)
            sse_diff = abs(model["inertia"] - reference["inertia"])
            sse_limit = max(1e-8, args.sse_relative_tolerance * abs(reference["inertia"]))
            correct = (model["iterations"] == reference["iterations"] and
                       centroid_diff <= args.centroid_tolerance and sse_diff <= sse_limit)
            row["centroid_max_abs_diff"] = centroid_diff
            row["sse_abs_diff"] = sse_diff
            row["correctness_pass"] = str(correct).lower()
            row["serial_reference_seconds"] = reference["runtime"]
            if not correct:
                row["status"] = "incorrect"
                row["diagnostic"] = (f"iterations {model['iterations']} vs {reference['iterations']}; "
                                     f"centroid max diff {centroid_diff:g} (limit {args.centroid_tolerance:g}); "
                                     f"SSE diff {sse_diff:g} (limit {sse_limit:g})")
            elif mode == "serial":
                row["speedup"] = 1.0
                row["efficiency"] = 1.0
            else:
                speedup = reference["runtime"] / model["runtime"] if model["runtime"] > 0 else math.inf
                row["speedup"] = speedup
                row["efficiency"] = speedup / (ranks * threads)
        return row, model
    except subprocess.TimeoutExpired as exc:
        row["diagnostic"] = f"timed out after {args.timeout_seconds}s; {str(exc)}"
    except FileNotFoundError as exc:
        row["diagnostic"] = str(exc)
    except (OSError, ValueError, ZeroDivisionError) as exc:
        row["diagnostic"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["wall_seconds_including_launcher"] = time.perf_counter() - start
    return row, None


def reproducibility_check(rows: list[dict], models: dict, args: argparse.Namespace) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            key = (row["experiment"], row["mode"], row["mpi_ranks"], row["threads_per_rank"], row["points"])
            grouped[key].append(row)
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        first = models[id(group[0])]
        same = True
        for row in group[1:]:
            current = models[id(row)]
            if (current["iterations"] != first["iterations"] or
                abs(current["inertia"] - first["inertia"]) > max(1e-8, args.sse_relative_tolerance * abs(first["inertia"])) or
                len(current["centroids"]) != len(first["centroids"]) or
                max(abs(a-b) for a,b in zip(current["centroids"], first["centroids"])) > args.centroid_tolerance):
                same = False
        for row in group:
            row["reproducibility_pass"] = str(same).lower()
            if not same:
                row["status"] = "nonreproducible"
                row["speedup"] = ""
                row["efficiency"] = ""
                row["diagnostic"] = (row["diagnostic"] + "; " if row["diagnostic"] else "") + \
                                     f"model varied across repetitions for {key}"


RAW_FIELDS = ["experiment", "repetition", "mode", "mpi_ranks", "threads_per_rank", "total_workers",
              "points", "dimensions", "clusters", "seed", "max_iterations_config", "tolerance_config",
              "runtime_seconds", "wall_seconds_including_launcher", "iterations", "inertia_sse",
              "observed_threads", "centroid_max_abs_diff", "sse_abs_diff", "correctness_pass",
              "reproducibility_pass", "serial_reference_seconds", "speedup", "efficiency",
              "exit_code", "status", "command", "diagnostic"]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["experiment"], row["mode"], row["mpi_ranks"], row["threads_per_rank"],
               row["total_workers"], row["points"], row["dimensions"], row["clusters"], row["seed"])
        groups[key].append(row)
    serial_means = {}
    for key, group in groups.items():
        if key[1] == "serial":
            good = [float(r["runtime_seconds"]) for r in group if r["status"] == "ok"]
            if good:
                serial_means[(key[0], key[5], key[6], key[7], key[8])] = statistics.mean(good)
    summary = []
    for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
        good = [r for r in group if r["status"] == "ok"]
        failed = len(group) - len(good)
        runtimes = [float(r["runtime_seconds"]) for r in good]
        mean_runtime = statistics.mean(runtimes) if runtimes else ""
        workers = key[4]
        ref = serial_means.get((key[0], key[5], key[6], key[7], key[8]))
        valid_summary = failed == 0 and bool(good) and ref is not None
        speedup = (ref / mean_runtime if key[1] != "serial" and valid_summary and mean_runtime else
                   (1.0 if key[1] == "serial" and valid_summary else ""))
        efficiency = (speedup / workers if speedup != "" and workers else "")
        iters = [int(r["iterations"]) for r in good]
        sses = [float(r["inertia_sse"]) for r in good]
        summary.append({
            "experiment": key[0], "mode": key[1], "mpi_ranks": key[2], "threads_per_rank": key[3],
            "total_workers": workers, "points": key[5], "dimensions": key[6], "clusters": key[7],
            "seed": key[8], "repetitions_requested": len(group), "successful_runs": len(good),
            "failed_runs": failed, "status": "OK" if valid_summary else "FAILED",
            "mean_runtime_seconds": mean_runtime,
            "runtime_stddev_seconds": statistics.stdev(runtimes) if len(runtimes) > 1 else (0.0 if runtimes else ""),
            "mean_iterations": statistics.mean(iters) if iters else "",
            "mean_inertia_sse": statistics.mean(sses) if sses else "",
            "mean_speedup": speedup, "parallel_efficiency": efficiency,
        })
    return summary


def write_markdown_summary(path: Path, rows: list[dict]) -> None:
    headers = ["Experiment", "Mode", "Ranks × threads", "Workers", "Points", "Mean time (s)",
               "Speedup", "Efficiency", "Iterations", "SSE", "Status", "Failed runs"]
    lines = ["# Phase 6 Benchmark Summary", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        value = lambda key: "" if row[key] == "" else str(row[key])
        lines.append("| " + " | ".join([
            row["experiment"], row["mode"], f"{row['mpi_ranks']} × {row['threads_per_rank']}",
            str(row["total_workers"]), str(row["points"]), value("mean_runtime_seconds"),
            value("mean_speedup"), value("parallel_efficiency"), value("mean_iterations"),
            value("mean_inertia_sse"), row["status"], str(row["failed_runs"]),
        ]) + " |")
    lines.extend(["", "Speedup = mean serial runtime / mean parallel runtime for the same dataset. ",
                  "Efficiency = speedup / total workers. Rows with failures are marked FAILED and have no speedup/efficiency."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SUMMARY_FIELDS = ["experiment", "mode", "mpi_ranks", "threads_per_rank", "total_workers", "points",
                  "dimensions", "clusters", "seed", "repetitions_requested", "successful_runs", "failed_runs",
                  "status", "mean_runtime_seconds", "runtime_stddev_seconds", "mean_iterations",
                  "mean_inertia_sse", "mean_speedup", "parallel_efficiency"]


def main() -> int:
    args = parse_args()
    serial_path, mpi_path = Path(args.serial_exe), Path(args.mpi_exe)
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"phase6_{now}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    models: dict[int, dict] = {}
    benchmark_plans = [("strong", args.strong_points, STRONG_CONFIGS)]
    if not args.skip_weak:
        benchmark_plans.append(("weak", args.weak_base_points, WEAK_CONFIGS))

    for experiment, base_points, configs in benchmark_plans:
        for repetition in range(1, args.repetitions + 1):
            if experiment == "strong":
                point_counts = {base_points}
            else:
                point_counts = {base_points * ranks * threads for ranks, threads in configs}
            baselines = {}
            for points in sorted(point_counts):
                row, model = run_one(args, experiment, "serial", 1, 1, points, repetition, None)
                rows.append(row)
                if model is not None and row["status"] == "ok":
                    models[id(row)] = model
                    row["correctness_pass"] = "true"
                    row["serial_reference_seconds"] = model["runtime"]
                    row["speedup"] = 1.0
                    row["efficiency"] = 1.0
                    baselines[points] = model
                else:
                    baselines[points] = None
            for ranks, threads in configs:
                points = base_points if experiment == "strong" else base_points * ranks * threads
                reference = baselines.get(points)
                row, model = run_one(args, experiment,
                                     "mpi-only" if threads == 1 else "hybrid",
                                     ranks, threads, points, repetition, reference)
                rows.append(row)
                if model is not None:
                    models[id(row)] = model
                print(f"{experiment} rep={repetition}/{args.repetitions} {ranks}x{threads} "
                      f"N={points}: {row['status']} runtime={row['runtime_seconds']}s", flush=True)

    reproducibility_check(rows, models, args)
    summary = create_summary(rows)
    write_csv(output_dir / "raw_results.csv", RAW_FIELDS, rows)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary)
    write_markdown_summary(output_dir / "summary.md", summary)

    failed = sum(1 for row in rows if row["status"] != "ok")
    print(f"\nRaw CSV: {output_dir / 'raw_results.csv'}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Runs: {len(rows)}; failed/incorrect/nonreproducible: {failed}")
    for row in summary:
        print(f"{row['experiment']:6} {row['mode']:8} {row['mpi_ranks']}x{row['threads_per_rank']} "
              f"N={row['points']:<8} mean={row['mean_runtime_seconds']}s "
              f"speedup={row['mean_speedup']} efficiency={row['parallel_efficiency']} status={row['status']}")
    for row in rows:
        if row["status"] != "ok":
            print(f"FAILED: {row['experiment']} rep={row['repetition']} {row['mode']} "
                  f"{row['mpi_ranks']}x{row['threads_per_rank']} N={row['points']}: {row['diagnostic']}",
                  file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
