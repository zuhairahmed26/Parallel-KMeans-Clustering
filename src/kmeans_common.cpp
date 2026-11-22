#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

double squared_distance(const double* first, const double* second, std::size_t dimensions) {
    double sum = 0.0;
    for (std::size_t d = 0; d < dimensions; ++d) {
        const double difference = first[d] - second[d];
        sum += difference * difference;
    }
    return sum;
}

std::vector<double> initialize_kmeans_plus_plus(const Dataset& data,
                                                std::size_t clusters,
                                                std::uint64_t seed) {
    if (clusters == 0 || clusters > data.point_count)
        throw std::invalid_argument("clusters must be in [1, points]");

    std::mt19937_64 random(seed);
    std::uniform_int_distribution<std::size_t> first_pick(0, data.point_count - 1);
    std::vector<double> centroids(clusters * data.dimensions);
    std::vector<double> nearest(data.point_count, std::numeric_limits<double>::infinity());

    std::size_t chosen = first_pick(random);
    for (std::size_t c = 0; c < clusters; ++c) {
        std::copy_n(data.values.data() + chosen * data.dimensions, data.dimensions,
                    centroids.data() + c * data.dimensions);

        double total = 0.0;
        for (std::size_t i = 0; i < data.point_count; ++i) {
            const double distance = squared_distance(data.values.data() + i * data.dimensions,
                                                     centroids.data() + c * data.dimensions,
                                                     data.dimensions);
            nearest[i] = std::min(nearest[i], distance);
            total += nearest[i];
        }
        if (c + 1 == clusters) break;

        if (total <= 0.0 || !std::isfinite(total)) {
            // Identical points have no distance-weighted choice; choose a reproducible point.
            chosen = (chosen + 1) % data.point_count;
            continue;
        }
        std::uniform_real_distribution<double> weighted_pick(0.0, total);
        const double target = weighted_pick(random);
        double cumulative = 0.0;
        chosen = data.point_count - 1;
        for (std::size_t i = 0; i < data.point_count; ++i) {
            cumulative += nearest[i];
            if (cumulative >= target) { chosen = i; break; }
        }
    }
    return centroids;
}
