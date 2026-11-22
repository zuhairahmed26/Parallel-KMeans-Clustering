# Parallel K-Means Clustering

This PDC project contains a sequential C++17 K-Means baseline and a hybrid MPI + OpenMP implementation. The serial executable is `kmeans`; `kmeans_mpi` uses MPI ranks for distributed memory and OpenMP threads for shared-memory point assignment within each rank.

## What K-Means does

K-Means groups points into `K` clusters. It repeatedly assigns each point to its closest centroid and replaces each centroid with the mean of its assigned points. Iteration stops when the largest centroid movement is at most the configured tolerance, or when the iteration limit is reached. Squared Euclidean distance avoids square roots during nearest-centroid comparisons.

## Project structure

- `src/dataset.cpp`: reproducible Gaussian blobs and their generating centers.
- `src/kmeans_common.cpp`: squared distance and seeded K-Means++ initialization.
- `src/kmeans_serial.cpp`: sequential assignment, centroid update, convergence, assignments and inertia (SSE).
- `src/kmeans_mpi.cpp`: MPI distribution/reductions and OpenMP local assignment with thread-private accumulators.
- `src/main.cpp`: serial CLI and result reporting.
- `src/main_mpi.cpp`: MPI startup, thread configuration, and hybrid result reporting.
- `include/`: interfaces for the dataset, CLI and K-Means implementations.
- `CMakeLists.txt`: builds serial independently and adds the MPI + OpenMP executable when those dependencies are found.
- `scripts/benchmark.py` and `scripts/plot_results.py`: repeatable benchmark orchestration and CSV-based plots.
- `docs/`: final technical report and demonstration/viva guide.
- `results/`: locally generated CSV summaries and PNG plots; output files are ignored by Git and are not overwritten by a new benchmark run.

```text
parallel-kmeans/
├── include/                 C++ interfaces
├── src/                     serial, dataset, CLI, MPI/OpenMP implementations
├── scripts/                 benchmark runner and Matplotlib plotter
├── docs/                    technical report and demonstration/viva guide
├── results/                 generated CSV and PNG outputs (local, ignored)
├── output/pdf/               submission-ready PDF copies
├── CMakeLists.txt
└── README.md
```

Points and centroids use contiguous row-major `double` arrays. Empty clusters retain their previous centroid to avoid division by zero. Synthetic points are generated round-robin across known Gaussian centers.

## Build the serial version on Windows

Requirements: CMake 3.20+ and a C++17 compiler. With the verified MSYS2 UCRT64 compiler:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build --target kmeans --config Release
```

The MPI + OpenMP target is built only when CMake finds both libraries; a missing MPI/OpenMP installation does not prevent the serial target from building.

## Build in Ubuntu on WSL2

The verified environment is Ubuntu on WSL2 with GCC, CMake, Open MPI and GCC's OpenMP runtime. If needed, install them inside Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake openmpi-bin libopenmpi-dev
```

Use the Windows checkout as the source and a Linux-home build directory so generated build files stay out of the Windows project:

```bash
cmake -S "/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project" -B "$HOME/parallel-kmeans-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/parallel-kmeans-build" -j
```

For a serial-only build, configure a separate directory with `-DBUILD_MPI=OFF`:

```bash
cmake -S "/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project" \
  -B "$HOME/parallel-kmeans-serial" -DBUILD_MPI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/parallel-kmeans-serial" --target kmeans -j
```

The CMake MPI target links `MPI::MPI_CXX` and `OpenMP::OpenMP_CXX`. The serial target does not link OpenMP.

## Run the serial baseline

On Windows:

```powershell
.\build\kmeans.exe --points 10000 --dimensions 2 --clusters 3 --seed 42
```

In Ubuntu, use `$HOME/parallel-kmeans-build/kmeans` with the same options. Available data and algorithm options are `--points`, `--dimensions`, `--clusters`, `--max-iterations`, `--tolerance`, `--seed` and `--stddev`. Use `--help` for defaults. The serial executable remains single-threaded and rejects `--threads`.

## Run hybrid MPI + OpenMP K-Means

```bash
mpirun -n 2 "$HOME/parallel-kmeans-build/kmeans_mpi" --points 10003 --dimensions 32 --clusters 5 --threads 4 --seed 42
```

This starts 2 MPI ranks and requests 4 OpenMP threads per rank, for 8 workers in total. The program reports the observed OpenMP team size, MPI ranks, iterations, runtime, inertia and final centroids.

### Thread configuration and precedence

1. `--threads N` explicitly sets the OpenMP team size per rank and takes precedence.
2. Without `--threads`, `OMP_NUM_THREADS` configures the OpenMP runtime.
3. If neither is set, the runtime chooses its default.

Dynamic team shrinking is disabled so the selected setting is used consistently. Examples:

```bash
# CLI thread setting overrides OMP_NUM_THREADS.
mpirun -n 2 "$HOME/parallel-kmeans-build/kmeans_mpi" --points 10003 --dimensions 32 --clusters 5 --threads 4 --seed 42

# Environment setting is used when --threads is omitted.
OMP_NUM_THREADS=4 mpirun -n 2 "$HOME/parallel-kmeans-build/kmeans_mpi" --points 10003 --dimensions 32 --clusters 5 --seed 42

# MPI-only comparison: one thread per rank.
mpirun -n 4 "$HOME/parallel-kmeans-build/kmeans_mpi" --points 10003 --dimensions 32 --clusters 5 --threads 1 --seed 42
```

`kmeans` is the serial baseline. `kmeans_mpi --threads 1` is MPI-only computation. `kmeans_mpi --threads N` with `N > 1` uses both MPI and OpenMP.

## Correctness validation

The benchmark runner also acts as an integration correctness check. This compact command uses an uneven point count and tests serial, MPI-only, and all default hybrid configurations against matching seeded serial references:

```bash
PROJECT="/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project"
python3 -B "$PROJECT/scripts/benchmark.py" \
  --serial-exe "$HOME/parallel-kmeans-build/kmeans" \
  --mpi-exe "$HOME/parallel-kmeans-build/kmeans_mpi" \
  --output-root /tmp/parallel-kmeans-check \
  --strong-points 10003 --dimensions 8 --clusters 3 \
  --max-iterations 100 --tolerance 0.001 --seed 42 \
  --repetitions 1 --skip-weak
```

The runner compares convergence state, iteration count, final centroids (absolute tolerance `1e-5`) and SSE (relative tolerance `1e-6`, with a `1e-8` floor). A non-zero exit status means at least one run failed, was incorrect, or did not reproduce across repetitions. The full Phase 7 verification used the same runner with two repetitions and 100,000-point strong scaling; its raw CSV is retained in the results folder described below.

## How MPI and OpenMP work together

- An **MPI rank** is a separate process with its own memory. The dataset is divided between ranks. `MPI_Scatterv` sends each rank its point slice, including uneven partitions. Rank 0 selects K-Means++ initial centroids and `MPI_Bcast` gives those same centroids to all ranks.
- An **OpenMP thread** is a worker inside one rank. Threads share that rank's local points and centroids, unlike MPI ranks, which have separate memories.
- Each rank enters an OpenMP `parallel` region. `omp for schedule(static)` divides point indices into predictable, near-equal chunks. Each point has roughly the same K-by-D distance work, so static scheduling avoids extra dynamic scheduling coordination.
- For each point, its thread finds the nearest centroid using squared Euclidean distance and writes that point's assignment. It also updates only its own K-by-D cluster-sum buffer and K-count buffer.
- **Thread-local buffers prevent race conditions.** A race would occur if two threads simultaneously modified the same cluster sum or count. Here, each thread writes to a private buffer, so the inner loop needs no locks or atomics.
- At the end of the parallel region, an implicit barrier waits for all threads to finish. The rank's main thread then merges private sums and counts into rank-level totals.
- The buffers are contiguous slices per thread. Their storage is proportional to `threads × clusters × dimensions`; very large values can make this temporary memory significant. We do not add cache-line padding, keeping the beginner-friendly layout simple.
- **`MPI_Allreduce` happens after the OpenMP region**, on the main thread. It sums the rank-level partials and gives the same global totals to every rank. Each rank computes matching centroids and takes the same convergence decision before the next iteration.

For example, 2 ranks × 4 threads uses two dataset partitions and four local workers per partition. 1 rank × 8 threads uses one partition and eight local workers. Both have eight workers, but the first distributes memory/work across processes and performs MPI communication. More ranks can increase communication and data partitioning overhead.

MPI calls use `MPI_THREAD_FUNNELED`: MPI is called only by the main thread, while OpenMP threads perform local calculations. The program does not request `MPI_THREAD_MULTIPLE`.

The reported MPI runtime starts after initial centroids and point distribution. It includes iterative assignment/reduction work and final inertia, and reports the maximum elapsed rank time. Dataset generation and initial distribution are outside the timer, as they are outside the serial algorithm timer.

## Final review scope and current limits

- Dataset generation and K-Means++ initialization run on rank 0. The root temporarily holds the full synthetic dataset before `MPI_Scatterv` and its own local slice afterward.
- MPI uses classic `int` element counts/displacements for `MPI_Scatterv`; configurations whose point or centroid element counts exceed `INT_MAX` are rejected before allocation/distribution.
- Each rank's thread-private sum storage is proportional to `threads × clusters × dimensions`.
- Assignments remain local to each rank in the normal CLI run; the program does not gather or write all labels. A correctness harness can gather them for comparisons.
- The program reports runtime, iterations, convergence, MPI ranks and OpenMP threads per rank. The benchmark runner records total workers, speedup, efficiency, SSE and convergence status in CSV.
- If a rank encounters an unexpected exception while MPI work is active, the program reports it and calls `MPI_Abort` so other ranks do not wait forever in a collective. Normal success and coordinated input-error paths finalize MPI normally.

## Troubleshooting

- If CMake cannot find OpenMP, verify that the compiler's OpenMP runtime/development support is installed, then rerun CMake configuration.
- `--threads` must be a positive integer. Zero, negative and non-integer values are rejected. The serial executable does not accept this option.
- If fewer threads than requested are observed, check runtime/system thread limits and verify the command's `OMP_NUM_THREADS` or `--threads` setting. CLI takes precedence.
- Avoid oversubscribing the machine in comparisons: total requested workers are MPI ranks multiplied by threads per rank.

## Phase 6 benchmarking

Benchmark automation is separate from the K-Means executables. The runner uses Python's standard library. Plotting requires Matplotlib (`scripts/requirements.txt`) and can run in a project-local virtual environment. If Ubuntu's `python3 -m venv` reports that `ensurepip` is missing, the matching `python3-venv` package must first be made available by the system administrator; install Matplotlib only inside the virtual environment.

Build the serial and hybrid executables as shown above, then run from Ubuntu:

```bash
PROJECT="/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project"
python3 -B "$PROJECT/scripts/benchmark.py" \
  --serial-exe "$HOME/parallel-kmeans-build/kmeans" \
  --mpi-exe "$HOME/parallel-kmeans-build/kmeans_mpi" \
  --output-root "$PROJECT/results" \
  --strong-points 100000 --weak-base-points 20000 \
  --dimensions 32 --clusters 5 --repetitions 3 --seed 42
```

Point counts, dimensions, clusters, seed, max iterations, tolerance, repetitions, timeout, and executable paths are configurable. The default strong suite uses serial plus 1×1, 1×2, 1×4, 2×1, 2×2, 2×4, 4×1, 4×2 and 4×4 (ranks × threads). Strong runs keep N fixed. The weak suite uses 1×1, 2×1, 2×2 and 4×2; N scales as base points × total workers (1, 2, 4 and 8 workers), while D and K remain fixed. `--skip-weak` runs only the strong suite.

Each invocation creates a timestamped folder under the output root with:

- `raw_results.csv`: one row for every attempt, including command, return code, diagnostic, correctness, reproducibility and convergence status. Failed, incorrect, unvalidated or non-reproducible runs remain visible and make the script exit non-zero.
- `summary.csv` and `summary.md`: average runtime, runtime standard deviation, average iterations/SSE, convergence state, success/failure counts, speedup and efficiency per configuration. Configurations with any failed or invalid repetition have no summary speedup/efficiency.
- Plot files generated by the plotting script below.

Run plots for a completed benchmark directory:

```bash
python3 -B "$PROJECT/scripts/plot_results.py" \
  "$PROJECT/results/<timestamp-folder>/summary.csv" \
  --output-dir "$PROJECT/results/<timestamp-folder>"
```

The plots cover runtime comparison, strong-scaling runtime, speedup, efficiency and (when weak data exists) weak-scaling runtime. They only plot fully successful summary rows.

### Metrics and timing

Speedup is `mean serial runtime / mean parallel runtime` for the same dataset parameters. Efficiency is `speedup / total workers`, where total workers means MPI ranks × OpenMP threads per rank. MPI-only runs use one thread per rank. A serial run has one worker.

Both executables report algorithm runtime, not Python/MPI launcher wall time. Serial timing includes assignment/update iterations and final assignment/inertia; it excludes synthetic data generation and K-Means++ initialization. MPI timing covers the same iterative and final-inertia work, plus thread-buffer merging and MPI reductions; it reports the slowest rank. It starts after data and centroids have been distributed, excluding dataset generation and initial distribution. `raw_results.csv` also records launcher wall time for context, but speedup uses application-reported runtime.

Before accepting a parallel run, the runner requires matching iteration count and convergence state, maximum centroid difference ≤ `1e-5`, and SSE difference ≤ `max(1e-8, 1e-6 × |serial SSE|)`. Repeated runs use the same seed and are checked for model reproducibility; runtime is expected to vary. A matching run that reached its iteration cap is recorded as `converged=false`, so a result can be correct relative to serial while still being unsuitable for a converged weak-scaling claim. Larger N can require substantial time and memory because rank 0 generates and holds the full dataset.

### Local plotting environment

From the project root in Ubuntu, create a local environment and install the plotting requirement there:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
```

The `.venv/` directory is ignored by Git. If virtual-environment support is missing from the Ubuntu Python installation, resolve that prerequisite before creating the environment. The Python package install itself is confined to `.venv`.

### Measured results

The following are measured application runtimes from Ubuntu on WSL2, using the actual program output. Strong scaling used N=100,000, D=32, K=5, seed 42, tolerance 0.001, a 500-iteration cap, and two repetitions. The serial mean was 0.036871 seconds (2 iterations, converged, SSE 3,202,258.706455).

| Mode | MPI ranks × threads/rank | Total workers | Mean runtime (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| MPI-only | 1 × 1 | 1 | 0.031660 | 1.165 | 1.165 |
| MPI-only | 2 × 1 | 2 | 0.016560 | 2.226 | 1.113 |
| MPI-only | 4 × 1 | 4 | 0.013437 | 2.744 | 0.686 |
| Hybrid | 1 × 2 | 2 | 0.023412 | 1.575 | 0.787 |
| Hybrid | 1 × 4 | 4 | 0.026030 | 1.416 | 0.354 |
| Hybrid | 2 × 2 | 4 | 0.016420 | 2.246 | 0.561 |
| Hybrid | 2 × 4 | 8 | 0.035668 | 1.034 | 0.129 |
| Hybrid | 4 × 2 | 8 | 0.019483 | 1.892 | 0.237 |
| Hybrid | 4 × 4 | 16 | 0.064344 | 0.573 | 0.036 |

The 16-worker configuration was slower than serial. The WSL environment reported 8 available logical processors (`nproc`), so that configuration oversubscribed the reported capacity. The implementation also creates and merges per-thread accumulator buffers and performs global reductions each iteration; the benchmark has no profiling data that separates these costs.

The revised weak-scaling run kept 20,000 points per worker, D=32, K=5, seed 42 and tolerance 0.001. It raised the maximum to 500 iterations; the 160,000-point, 8-worker case converged in 172 iterations. All configurations converged and matched the serial baseline in two repetitions, but iteration counts ranged from 2 to 172, so the runtime trend is not a clean fixed-work-per-worker comparison.

| MPI ranks × threads/rank | Workers | Points | Mean runtime (s) | Iterations | Converged | SSE |
|---:|---:|---:|---:|---:|:---:|---:|
| 1 × 1 (MPI-only) | 1 | 20,000 | 0.006269 | 2 | yes | 640,245.213463 |
| 2 × 1 (MPI-only) | 2 | 40,000 | 0.007226 | 2 | yes | 1,281,773.957666 |
| 2 × 2 (hybrid) | 4 | 80,000 | 0.023430 | 3 | yes | 2,561,947.843404 |
| 4 × 2 (hybrid) | 8 | 160,000 | 0.763506 | 172 | yes | 23,711,673.702115 |

All plots, CSV files and the full analysis are in the timestamped `results/phase6_20261123_020832_500940/` folder. The first Phase 6 run is preserved in `results/phase6_20261123_015106_878239/`; its 8-worker weak case reached the original 100-iteration cap and is not used as the converged weak-scaling result.

## Final submission documents

- [Technical report](docs/technical_report.md)
- [Demonstration and viva guide](docs/demonstration_viva_guide.md)
- PDF copies are created under `output/pdf/` for printing and submission.

The report includes F1–F7 implementation evidence, the measured tables and linked PNG figures. The viva guide contains the full workflow, command sequence, performance explanations and more than 25 questions with answers.

## Roadmap

Serial K-Means, MPI distribution/reduction, OpenMP assignment, benchmarking and report preparation are complete. Future improvements include stronger weak-scaling controls, larger sustained workloads, richer memory/CPU profiling and optionally gathering final assignments for external inspection. These are not part of the current implementation.
