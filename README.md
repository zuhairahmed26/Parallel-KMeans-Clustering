# Parallel K-Means Clustering

This project is being built in phases for a PDC course. The current phase provides a serial C++17 K-Means baseline and synthetic Gaussian data; MPI and hybrid execution are intentionally not part of this version yet.

## What K-Means does

K-Means groups points into `K` clusters. It repeatedly assigns each point to its closest centroid and replaces each centroid with the mean of the points assigned to it. Iteration stops when the largest centroid movement is at most the configured tolerance, or when the iteration limit is reached.

The distance calculation uses squared Euclidean distance. The square root is unnecessary when comparing distances and is only used when comparing centroid movement with the tolerance.

## Current architecture

- `dataset.cpp`: creates reproducible Gaussian blobs and retains their generating centers as ground truth.
- `kmeans_common.cpp`: provides squared distance and seeded K-Means++ initialization.
- `kmeans_serial.cpp`: serial assignment, centroid sums and counts, updates, convergence, final assignments and sum of squared errors (inertia).
- `cli.cpp` / `main.cpp`: input validation, execution and a concise result report.
- Data coordinates and centroids use contiguous row-major `double` arrays.

Empty clusters keep their previous centroid. This is deterministic and avoids division by zero. The generator distributes points round-robin among known centers, then adds independent Gaussian noise.

## Build

Requirements: CMake 3.20+ and a C++17 compiler. From a configured compiler shell:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build --config Release
```

The compiler above was observed in the development environment as MSYS2 GCC 13.2.0. The commands select the installed compiler explicitly. If using Visual Studio, configure CMake from a Visual Studio Developer PowerShell instead.

## Run

```powershell
.\build\kmeans.exe --points 10000 --dimensions 2 --clusters 3 --seed 42
```

Options include `--points`, `--dimensions`, `--clusters`, `--max-iterations`, `--tolerance`, `--seed`, and `--stddev`. Use `--help` to see defaults and descriptions. `--stddev` controls Gaussian spread.

## Roadmap

The next phases will add MPI distribution and reductions, OpenMP assignment and thread-local accumulation, then correctness comparisons and automated strong/weak scaling benchmarks. No performance claims are made until those experiments are run.
