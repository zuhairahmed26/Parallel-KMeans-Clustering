#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Points are stored consecutively: coordinate d of point i is at i * dimensions + d.
struct Dataset {
    std::size_t point_count = 0;
    std::size_t dimensions = 0;
    std::vector<double> values;
    std::vector<double> true_centers; // Synthetic generator ground truth, K x D.
};

Dataset generate_gaussian_blobs(std::size_t point_count,
                                std::size_t dimensions,
                                std::size_t cluster_count,
                                std::uint64_t seed,
                                double standard_deviation = 1.0);
