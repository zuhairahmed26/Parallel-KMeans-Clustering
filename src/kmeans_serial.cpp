#include "kmeans_serial.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

KMeansResult run_serial_kmeans(const Dataset& data,
                               const KMeansOptions& options,
                               const std::vector<double>& initial_centroids) {
    if (data.point_count == 0 || data.dimensions == 0 || data.values.size() != data.point_count * data.dimensions)
        throw std::invalid_argument("dataset shape is invalid");
    if (options.clusters == 0 || options.clusters > data.point_count)
        throw std::invalid_argument("clusters must be in [1, points]");
    if (options.max_iterations == 0 || options.tolerance < 0.0 || !std::isfinite(options.tolerance))
        throw std::invalid_argument("maximum iterations must be positive and tolerance non-negative");
    if (initial_centroids.size() != options.clusters * data.dimensions)
        throw std::invalid_argument("initial centroid array has the wrong size");

    KMeansResult result;
    result.centroids = initial_centroids;
    result.assignments.resize(data.point_count, -1);
    std::vector<double> sums(options.clusters * data.dimensions);
    std::vector<std::size_t> counts(options.clusters);
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        // Assignment: each point chooses the closest centroid using squared Euclidean distance.
        for (std::size_t i = 0; i < data.point_count; ++i) {
            const double* point = data.values.data() + i * data.dimensions;
            std::size_t best_cluster = 0;
            double best_distance = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < options.clusters; ++c) {
                const double distance = squared_distance(point, result.centroids.data() + c * data.dimensions,
                                                         data.dimensions);
                if (distance < best_distance) { best_distance = distance; best_cluster = c; }
            }
            result.assignments[i] = static_cast<int>(best_cluster);
            ++counts[best_cluster];
            for (std::size_t d = 0; d < data.dimensions; ++d)
                sums[best_cluster * data.dimensions + d] += point[d];
        }

        // Update: average each non-empty cluster. Empty clusters retain their prior centroid.
        double maximum_shift_squared = 0.0;
        for (std::size_t c = 0; c < options.clusters; ++c) {
            if (counts[c] == 0) continue;
            std::vector<double> updated(data.dimensions);
            for (std::size_t d = 0; d < data.dimensions; ++d)
                updated[d] = sums[c * data.dimensions + d] / static_cast<double>(counts[c]);
            maximum_shift_squared = std::max(maximum_shift_squared,
                squared_distance(result.centroids.data() + c * data.dimensions, updated.data(), data.dimensions));
            std::copy(updated.begin(), updated.end(), result.centroids.begin() + c * data.dimensions);
        }
        result.iterations = iteration + 1;
        if (std::sqrt(maximum_shift_squared) <= options.tolerance) {
            result.converged = true;
            break;
        }
    }

    // Reassign against final centroids so assignments and inertia describe the returned model.
    result.inertia = 0.0;
    for (std::size_t i = 0; i < data.point_count; ++i) {
        const double* point = data.values.data() + i * data.dimensions;
        std::size_t best_cluster = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < options.clusters; ++c) {
            const double distance = squared_distance(point, result.centroids.data() + c * data.dimensions,
                                                     data.dimensions);
            if (distance < best_distance) { best_distance = distance; best_cluster = c; }
        }
        result.assignments[i] = static_cast<int>(best_cluster);
        result.inertia += best_distance;
    }
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
