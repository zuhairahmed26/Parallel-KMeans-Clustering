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
- The program currently reports runtime, iterations, MPI ranks and OpenMP threads per rank. Total worker count can be calculated as ranks × threads. Speedup, efficiency, CSV automation, plots and strong/weak scaling are intentionally deferred to Phase 6.
- If a rank encounters an unexpected exception while MPI work is active, the program reports it and calls `MPI_Abort` so other ranks do not wait forever in a collective. Normal success and coordinated input-error paths finalize MPI normally.

## Troubleshooting

- If CMake cannot find OpenMP, verify that the compiler's OpenMP runtime/development support is installed, then rerun CMake configuration.
- `--threads` must be a positive integer. Zero, negative and non-integer values are rejected. The serial executable does not accept this option.
- If fewer threads than requested are observed, check runtime/system thread limits and verify the command's `OMP_NUM_THREADS` or `--threads` setting. CLI takes precedence.
- Avoid oversubscribing the machine in comparisons: total requested workers are MPI ranks multiplied by threads per rank.

## Roadmap

Automated correctness and strong/weak scaling benchmarks are later phases. No benchmark or performance claim is made until those experiments are run.
