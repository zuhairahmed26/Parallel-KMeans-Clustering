#pragma once

#include "dataset.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct KMeansOptions {
    std::size_t clusters = 3;
    std::size_t max_iterations = 100;
    double tolerance = 1e-4;
    std::uint64_t seed = 42;
};

struct KMeansResult {
    std::vector<double> centroids; // K x D, contiguous.
    std::vector<int> assignments;  // One cluster index per point.
    std::size_t iterations = 0;
    double inertia = 0.0;          // Sum of squared point-to-centroid distances.
    double elapsed_seconds = 0.0;
    bool converged = false;
    int threads_used = 1;          // Actual OpenMP team size in the MPI version.
};

std::vector<double> initialize_kmeans_plus_plus(const Dataset& data,
                                                std::size_t clusters,
                                                std::uint64_t seed);

double squared_distance(const double* first, const double* second, std::size_t dimensions);
