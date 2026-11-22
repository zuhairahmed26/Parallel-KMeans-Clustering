# Parallel K-Means Demonstration and Viva Guide

This guide explains the project from program startup through the final clustering model. It is written to support a short live demonstration and to help students explain the implementation choices in a viva. The commands below target the verified Ubuntu-on-WSL2 environment and use the Windows project checkout as the source.

## 1. Project Workflow

The program starts by parsing command-line settings: point count, dimensions, cluster count, random seed, iteration cap, tolerance and, for the MPI executable, threads per rank. Invalid values produce an error rather than entering the algorithm.

The serial executable generates deterministic Gaussian blobs, chooses starting centroids with K-Means++, then repeatedly assigns points and updates centroids. The MPI executable initializes MPI, determines its rank and world size, and lets rank 0 generate the dataset and initialize the same K-Means++ model. It partitions and scatters points with `MPI_Scatterv`, broadcasts initial centroids, then performs local work on each rank. OpenMP threads share that rank's point partition, but each thread has its own cluster sums and counts. The main thread merges them, MPI combines the rank totals, and every rank updates the same centroid values. The loop stops after convergence or the iteration cap. The program then calculates final SSE and prints the result.

## 2. Serial Implementation

The serial executable is `kmeans`. `src/main.cpp` prepares data and starting centroids, then calls `run_serial_kmeans` in `src/kmeans_serial.cpp`.

Each iteration does four things:

1. Clear cluster sums and point counts.
2. For every point, calculate squared distance to each centroid and choose the nearest one.
3. Add the point's coordinates to that cluster's sum and increment its count.
4. Replace each non-empty centroid with its coordinate-wise mean.

After updating centroids, the algorithm measures the largest centroid movement. If it is no greater than the tolerance, it stops. Otherwise, it continues until `--max-iterations`. An empty cluster retains its previous centroid. The final inertia/SSE pass measures the sum of squared distances from every point to its nearest final centroid.

The serial version is the correctness reference. For a reproducible comparison, the serial and MPI commands must use the same point count, dimensions, K, seed, tolerance and maximum iterations.

## 3. MPI Implementation

### MPI initialization and rank responsibilities

`src/main_mpi.cpp` initializes MPI using `MPI_Init_thread` and asks for `MPI_THREAD_FUNNELED`. This means only the main thread may call MPI functions. Each MPI process has a rank number. Rank 0 is the root process; it generates the full dataset and computes the initial centroids. Other ranks wait in the same collective calls and later receive their local points.

### Dataset partition and `MPI_Scatterv`

With `P` ranks and `N` points, each rank gets `N / P` points. The first `N % P` ranks get one extra point. `MPI_Scatterv` uses arrays of counts and displacements to send each rank a possibly different amount of data. This is why the program handles sizes such as 10,003 points with 4 ranks.

### Centroid broadcast and local computation

Rank 0 selects K-Means++ centroids using the configured seed. `MPI_Bcast` sends that centroid array to all ranks so the local assignment steps all begin from one agreed model. Each rank then finds the nearest centroid for its own point slice.

### Global reduction and update

Each rank has cluster coordinate sums and point counts for its local data. `MPI_Allreduce` adds each rank's values and returns the global totals to every rank. The ranks divide global coordinate sums by global counts to calculate identical updated centroids. A count of zero means the cluster is empty, so its old centroid remains unchanged.

All ranks enter the collectives in the same order. They get the same global totals, compute the same maximum centroid shift, and therefore take the same continue/stop decision. A rank-local failure during MPI work calls `MPI_Abort` so peers do not hang indefinitely waiting in a collective.

## 4. OpenMP Implementation

OpenMP threads are workers inside one rank. They share the rank's point buffer and current centroids. The assignment code uses `#pragma omp parallel` and `#pragma omp for schedule(static)` to divide point indices into chunks. Static scheduling gives each thread a predictable share; each point performs a similar number of distance calculations.

Each thread writes only to its own slice of `thread_sums` and `thread_counts`. If all threads wrote to the same cluster count or sum, simultaneous updates could be lost or corrupt the result. Private slices avoid those races without locks in the point loop. When the parallel region ends, its implicit barrier guarantees that all threads are finished. The main thread then merges thread slices into rank totals and proceeds to MPI calls.

The private accumulator storage is proportional to `threads × clusters × dimensions`, plus the per-thread cluster counts. The code does not add cache-line padding, so false sharing is a possible optimization target, though it was not profiled in these experiments.

## 5. One Hybrid Iteration

For example, `mpirun -n 2 ... --threads 4` creates two ranks and requests four OpenMP threads in each rank, for eight total workers. Each rank owns a different point slice; its four threads cooperate on that slice.

Within each iteration:

1. Each rank clears its sums and counts.
2. OpenMP threads assign local points and update private accumulators.
3. The rank's main thread merges those private values.
4. MPI `Allreduce` combines sums and counts globally.
5. Each rank updates all centroids identically.
6. All ranks test the same maximum shift and either continue or stop.

MPI communication does not happen inside the OpenMP assignment region. This respects the requested `MPI_THREAD_FUNNELED` support level.

## 6. Performance Concepts

- **Runtime:** The measured elapsed time for the application work. The reported algorithm runtime excludes synthetic data generation and K-Means++ initialization. MPI timing also excludes initial point distribution and centroid broadcast; it includes iterative local work, merging, reductions, updates and final inertia, and reports the slowest rank.
- **Total workers:** MPI ranks multiplied by OpenMP threads per rank. MPI-only mode uses one thread per rank.
- **Speedup:** Serial runtime divided by parallel runtime for the same dataset parameters.
- **Parallel efficiency:** Speedup divided by total workers. Efficiency above one can appear in short/noisy measurements; it is not evidence of a universal hardware limit being exceeded.
- **Strong scaling:** Keep dataset size fixed and increase workers.
- **Weak scaling:** Increase dataset size in proportion to workers so points per worker stay roughly fixed.
- **Communication and synchronization overhead:** MPI collectives make ranks wait for global totals; OpenMP team setup and accumulator merging add work around the parallel loop.

The measured runtime is an algorithm-region comparison, not end-to-end startup time. Rank 0 also generates and temporarily holds the full dataset, which can become a memory bottleneck at larger sizes.

## 7. Demonstration Setup and Commands

Open Ubuntu on WSL2 and move to the project checkout. These commands compile into the Linux home directory, keeping build products separate from the Windows source tree:

```bash
PROJECT="/mnt/c/Fast Nuces/Semester 7/PDC/PDC Project"
cmake -S "$PROJECT" -B "$HOME/parallel-kmeans-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/parallel-kmeans-build" -j
```

### Run the serial baseline

```bash
"$HOME/parallel-kmeans-build/kmeans" \
  --points 10003 --dimensions 8 --clusters 3 --seed 42 \
  --tolerance 0.001 --max-iterations 100
```

### Run MPI-only with four ranks

```bash
mpirun --oversubscribe -n 4 "$HOME/parallel-kmeans-build/kmeans_mpi" \
  --points 10003 --dimensions 8 --clusters 3 --threads 1 --seed 42 \
  --tolerance 0.001 --max-iterations 100
```

### Run hybrid with two ranks and four threads per rank

```bash
mpirun --oversubscribe -n 2 "$HOME/parallel-kmeans-build/kmeans_mpi" \
  --points 10003 --dimensions 8 --clusters 3 --threads 4 --seed 42 \
  --tolerance 0.001 --max-iterations 100
```

Use identical K-Means options for the three commands. `--threads` applies only to `kmeans_mpi`. The CLI setting overrides `OMP_NUM_THREADS`.

### Run a correctness-checked benchmark sample

```bash
python3 -B "$PROJECT/scripts/benchmark.py" \
  --serial-exe "$HOME/parallel-kmeans-build/kmeans" \
  --mpi-exe "$HOME/parallel-kmeans-build/kmeans_mpi" \
  --output-root "$PROJECT/results" \
  --strong-points 100000 --dimensions 32 --clusters 5 \
  --max-iterations 500 --tolerance 0.001 --seed 42 --repetitions 2 --skip-weak
```

Every run creates a new timestamped directory. It does not overwrite earlier benchmark output.

## 8. What to Show During the Demonstration

1. Show the serial command and point out mode, iterations, convergence, runtime, SSE and centroid output.
2. Run the same parameters with four MPI ranks and one thread per rank. Point out the MPI rank count and `OpenMP threads per rank: 1`.
3. Run with two ranks and four threads. Point out that there are 8 workers total, made from two MPI processes with four local threads each.
4. Use N=10,003 to explain that rank counts can differ when N does not divide evenly by the number of ranks.
5. Show a benchmark summary row and explain that each speedup is compared with the serial runtime for the same dataset size. Show raw CSV status fields as evidence that a run completed and matched serial.
6. Show the strong runtime and speedup plots, then be honest that more workers were not always faster.
7. In weak scaling, point out that the 160,000-point case converged in 172 iterations after raising the cap to 500. It took much longer than smaller cases partly because iteration counts were not held fixed.

## 9. Measured Results to Remember

The verified strong-scaling set used N=100,000, D=32 and K=5, with seed 42, tolerance 0.001, 500 maximum iterations and two repetitions. Serial averaged 0.036871 seconds. The best measured runtime was 0.013437 seconds for four MPI ranks with one thread each. The 4-rank × 4-thread case used 16 workers while WSL reported 8 available logical processors; it averaged 0.064344 seconds, slower than serial. These are results from this environment and workload, not universal predictions.

In the follow-up weak run, each worker represented 20,000 points. MPI-only at 1 and 2 workers took 0.006269 and 0.007226 seconds; the 4-worker hybrid took 0.023430 seconds; the 8-worker hybrid took 0.763506 seconds. Every configuration converged, but the iteration counts were 2, 2, 3 and 172. That difference means the weak-scaling curve should be described as inconclusive for constant-time scaling.

## 10. Known Limitations and Defensible Decisions

- Synthetic Gaussian data are useful for reproducibility but do not represent every real dataset.
- Rank 0 creates and temporarily owns all points before distribution.
- Classic `MPI_Scatterv` count/displacement arguments are `int`; unsupported oversized inputs are rejected.
- The program does not gather every final label at rank 0 in the normal run.
- The algorithm may hit its configured iteration cap. The benchmark summary records convergence state.
- The timings exclude setup and initial distribution, and include different overhead components between serial and MPI execution as documented in the report.
- The weak-scaling run converged after increasing the iteration limit, but the number of iterations still varied substantially as N changed.
- Two repetitions are not enough to make broad performance claims. CPU pinning, cache/memory counters and detailed profiling are not part of this project.
- OpenMP is kept inside MPI ranks and MPI calls are restricted to the main thread to satisfy `MPI_THREAD_FUNNELED`.

## 11. Viva Questions and Answers

### Concepts

1. **What problem does K-Means solve?**  It groups numeric points into K clusters by repeatedly assigning each point to a nearest center and recomputing centers as means.
2. **Why do we need a serial implementation?**  It provides a simpler reference for checking the parallel outputs and measuring speedup.
3. **What does K represent?**  The number of clusters and centroids the algorithm will produce.
4. **What is a centroid?**  The coordinate-wise mean of the points assigned to one cluster.
5. **Why use squared Euclidean distance?**  It preserves distance ordering without computing a square root for every point-centroid comparison.
6. **What is K-Means++?**  A seeded initialization method that prefers selecting new centers far from already selected centers, often producing better starting separation than arbitrary picks.
7. **What does the random seed do?**  It makes synthetic data and centroid initialization reproducible for the same configuration.
8. **What is inertia or SSE?**  The sum of squared distances between each point and its nearest final centroid; lower values indicate tighter clusters for the same data and K.
9. **How is convergence detected?**  The code measures the largest centroid movement and stops when it is at most the tolerance.
10. **What happens if a cluster is empty?**  Its prior centroid is retained, avoiding division by zero and invalid values.

### MPI

11. **What is an MPI rank?**  A separate process with its own address space, identified by an integer rank.
12. **What is rank 0 responsible for?**  It generates the full synthetic dataset and selects initial centroids before distributing data/model state.
13. **Why divide the dataset among ranks?**  Each rank can perform assignment work on a smaller local point set, distributing memory and computation.
14. **How does the code support uneven N?**  It computes per-rank counts and offsets; the first remainder ranks receive one extra point.
15. **What does `MPI_Scatterv` do?**  It scatters varying counts of data to processes using count and displacement arrays.
16. **Why broadcast centroids?**  All ranks must begin each iteration with the same centroid values.
17. **What does `MPI_Allreduce` do here?**  It sums each rank's partial cluster sums/counts and returns global totals to every rank.
18. **Why use Allreduce instead of Reduce?**  Every rank needs the global values to compute identical centroids and convergence decisions without another broadcast.
19. **How do we avoid ranks getting out of sync?**  They call collectives in the same order and use the same global sums/counts and shift value.
20. **What is `MPI_THREAD_FUNNELED`?**  It allows only the main thread to call MPI; OpenMP worker threads do local computation but no MPI calls.

### OpenMP and Hybrid Integration

21. **What is an OpenMP thread?**  A worker inside one process that shares that process's memory.
22. **What does `schedule(static)` mean?**  The point loop is divided into predictable chunks assigned to threads ahead of execution.
23. **Why are sums and counts thread-local?**  Two threads updating the same value concurrently could lose updates; private slices avoid those races.
24. **When are thread-local values merged?**  After the parallel region ends and its implicit barrier ensures all threads have finished.
25. **Why not use a critical section for every point?**  It would serialize updates and add lock contention in the hot loop.
26. **How do MPI and OpenMP divide responsibilities?**  MPI partitions points across processes; OpenMP divides each process's local assignment loop across threads.
27. **What is total workers for 2 ranks × 4 threads?**  Eight workers: two MPI processes, each with four OpenMP threads.
28. **Why might hybrid run slower than MPI-only?**  This implementation adds OpenMP team coordination, thread-buffer clearing and merging, and MPI reductions. For short workloads those costs can outweigh the saved point work. The experiment did not profile individual costs.
29. **Why was 4 × 4 slow in this test?**  It requested 16 workers while WSL reported 8 available logical processors, so the run oversubscribed that reported capacity; thread and communication overhead also remain in the timed region.
30. **Why can floating-point results differ slightly?**  Parallel reductions add values in a different order from serial summation, and floating-point addition is not associative.

### Performance and Evaluation

31. **How is speedup calculated?**  Serial runtime divided by parallel runtime for the same dataset parameters.
32. **How is efficiency calculated?**  Speedup divided by total workers, defined as ranks × threads per rank.
33. **What is strong scaling?**  Keeping N fixed while increasing workers and measuring how runtime changes.
34. **What is weak scaling?**  Increasing N roughly in proportion to workers so the point count per worker stays similar.
35. **Why did the original 8-worker weak run need a follow-up?**  It hit the 100-iteration limit; raising the cap to 500 allowed it to converge in 172 iterations.
36. **Does that prove ideal weak scaling?**  No. The configurations needed 2 to 172 iterations, so the amount of algorithm work changed as well as the worker count.
37. **Why can measured efficiency exceed 1?**  The formula uses observed runtime, which can be affected by small sample noise and scheduling. It should be reported honestly, not interpreted as universal efficiency.
38. **What does the benchmark time include?**  It includes iterative algorithm work and final inertia; MPI time also includes thread merging/reductions but excludes generation, K-Means++ initialization and initial distribution.
39. **Why do plots use the CSV summary?**  It ensures figure values correspond to recorded attempts and calculated summary metrics rather than hand-entered numbers.
40. **What would you improve next?**  Use more repetitions and warm-up, collect profiling/affinity data, include end-to-end timings, and control initial centroids for weak scaling so iteration counts are more comparable.
