#include "dataset.h"

#include <cmath>
#include <limits>
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
    if (point_count > data.values.max_size() / dimensions ||
        cluster_count > data.true_centers.max_size() / dimensions)
        throw std::length_error("dataset dimensions exceed addressable vector storage");

    data.point_count = point_count;
    data.dimensions = dimensions;
    const std::size_t value_count = point_count * dimensions;
    const std::size_t center_value_count = cluster_count * dimensions;
    data.values.resize(value_count);
    data.true_centers.resize(center_value_count);

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
