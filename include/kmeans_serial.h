#pragma once

#include "kmeans.h"

KMeansResult run_serial_kmeans(const Dataset& data,
                               const KMeansOptions& options,
                               const std::vector<double>& initial_centroids);
