#include "dataset.h"

#include <cmath>
#include <random>
#include <stdexcept>

Dataset generate_gaussian_blobs(std::size_t point_count,
                                std::size_t dimensions,
                                std::size_t cluster_count,
                                std::uint64_t seed,
                                double standard_deviation) {
    if (point_count == 0 || dimensions == 0 || cluster_count == 0 || cluster_count > point_count)
        throw std::invalid_argument("points and dimensions must be positive; clusters must be in [1, points]");
    if (!(standard_deviation > 0.0) || !std::isfinite(standard_deviation))
        throw std::invalid_argument("standard deviation must be a positive finite number");

    Dataset data;
    data.point_count = point_count;
    data.dimensions = dimensions;
    data.values.resize(point_count * dimensions);
    data.true_centers.resize(cluster_count * dimensions);

    std::mt19937_64 random(seed);
    // Keep generated centers separated while remaining deterministic for a given seed.
    std::uniform_real_distribution<double> center_distribution(-10.0, 10.0);
    std::normal_distribution<double> noise(0.0, standard_deviation);
    for (double& coordinate : data.true_centers) coordinate = center_distribution(random);

    for (std::size_t i = 0; i < point_count; ++i) {
        const std::size_t cluster = i % cluster_count;
        for (std::size_t d = 0; d < dimensions; ++d) {
            data.values[i * dimensions + d] = data.true_centers[cluster * dimensions + d] + noise(random);
        }
    }
    return data;
}
