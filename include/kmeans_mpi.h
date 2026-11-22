#pragma once

#include "kmeans.h"

#include <mpi.h>

// Rank 0 supplies the full dataset. The function distributes points, uses
// OpenMP for rank-local assignment, and returns the final model on every rank.
KMeansResult run_mpi_kmeans(const Dataset& root_data,
                            const KMeansOptions& options,
                            int rank,
                            int world_size,
                            int thread_count,
                            MPI_Comm communicator = MPI_COMM_WORLD);
