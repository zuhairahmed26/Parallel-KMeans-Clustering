#pragma once

#include "kmeans.h"

#include <mpi.h>

// Rank 0 supplies the full dataset. The function distributes points, performs
// serial assignment on each rank, and returns the final model on every rank.
KMeansResult run_mpi_kmeans(const Dataset& root_data,
                            const KMeansOptions& options,
                            int rank,
                            int world_size,
                            MPI_Comm communicator = MPI_COMM_WORLD);
