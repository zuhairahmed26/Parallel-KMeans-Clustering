# Parallel K-Means Clustering

This project is being built in phases for a PDC course. It contains a serial C++17 K-Means baseline and a distributed-memory MPI version. The MPI assignment loop is intentionally serial within each rank; OpenMP hybrid parallelism is a later phase.

## What K-Means does

K-Means groups points into `K` clusters. It repeatedly assigns each point to its closest centroid and replaces each centroid with the mean of the points assigned to it. Iteration stops when the largest centroid movement is at most the configured tolerance, or when the iteration limit is reached.

The distance calculation uses squared Euclidean distance. The square root is unnecessary when comparing distances and is only used when comparing centroid movement with the tolerance.

## Current architecture

- `dataset.cpp`: creates reproducible Gaussian blobs and retains their generating centers as ground truth.
- `kmeans_common.cpp`: provides squared distance and seeded K-Means++ initialization.
- `kmeans_serial.cpp`: serial assignment, centroid sums and counts, updates, convergence, final assignments and sum of squared errors (inertia).
- `cli.cpp` / `main.cpp`: input validation, execution and a concise result report.
- `kmeans_mpi.cpp` / `main_mpi.cpp`: MPI distribution, serial local work, global reductions and the MPI result report.
- Data coordinates and centroids use contiguous row-major `double` arrays.

Empty clusters keep their previous centroid. This is deterministic and avoids division by zero. The generator distributes points round-robin among known centers, then adds independent Gaussian noise.

## Build the serial version on Windows

Requirements: CMake 3.20+ and a C++17 compiler. From a configured compiler shell:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build --config Release
```

The compiler above was observed in the development environment as MSYS2 GCC 13.2.0. The commands select the installed compiler explicitly. If using Visual Studio, configure CMake from a Visual Studio Developer PowerShell instead. CMake will still build the serial target if it cannot find MPI.

## Build in Ubuntu on WSL2

The MPI development environment used for this project is Ubuntu on WSL2 with GCC, CMake and Open MPI. Install the tools once inside Ubuntu if they are not present:

```bash
sudo apt update
sudo apt install build-essential cmake openmpi-bin libopenmpi-dev
```

From Ubuntu, configure using the Windows checkout as the source and a Linux-side build directory. Building under the Linux home directory avoids writing Linux build artifacts into the Windows project:

```bash
cmake -S "/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project" -B "$HOME/parallel-kmeans-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/parallel-kmeans-build" -j
```

To build only the MPI target, use `cmake --build "$HOME/parallel-kmeans-build" --target kmeans_mpi -j`.

## Run

```powershell
.\build\kmeans.exe --points 10000 --dimensions 2 --clusters 3 --seed 42
```

Options include `--points`, `--dimensions`, `--clusters`, `--max-iterations`, `--tolerance`, `--seed`, and `--stddev`. Use `--help` to see defaults and descriptions. `--stddev` controls Gaussian spread.

## Run MPI K-Means

Run from Ubuntu/WSL2 after building:

```bash
mpirun -n 3 "$HOME/parallel-kmeans-build/kmeans_mpi" --points 10003 --dimensions 32 --clusters 5 --seed 42
```

The MPI executable accepts the same data and algorithm options as the serial executable. Use the same options and seed with `kmeans` and `kmeans_mpi` to compare results. The report includes rank count, iterations, elapsed time, inertia (SSE), and centroids. This phase uses one serial assignment loop per MPI rank; it reports one OpenMP thread per rank because OpenMP is not integrated yet.

## How MPI works in this phase

- A **rank** is one independent process in the MPI job. Each rank has its own memory and its own local points.
- The dataset is divided so each rank computes on part of the points. The first `N % ranks` ranks receive one extra point, which balances uneven divisions.
- **`MPI_Scatterv`** sends each rank its point slice. The `v` variant takes per-rank counts and offsets, so ranks can receive different amounts.
- Rank 0 runs seeded K-Means++ once. **`MPI_Bcast`** sends those same starting centroids to every rank so the parallel run starts from the serial run's model.
- Each rank assigns its local points serially and accumulates local cluster sums and counts.
- **`MPI_Allreduce`** sums those partial values across all ranks and gives the global result to every rank. Every rank therefore computes identical new centroids and sees the same convergence decision.
- Empty clusters keep their previous centroid. This avoids dividing by zero. All ranks iterate together and stop at the same convergence check or iteration limit.

Each MPI rank reports only its local assignments internally; the public output reports the global centroids and inertia. The reported MPI runtime starts after initial centroids and points have been distributed. It includes iteration/reduction work and final inertia, and is the maximum elapsed time across ranks, which captures load imbalance. Dataset generation and initial distribution are outside this algorithm timing, as they are outside the serial K-Means timing.

## Roadmap

The next phase will add OpenMP assignment and thread-local accumulation. Later work will add correctness tests and automated strong/weak scaling benchmarks. No performance claims are made until those experiments are run.
