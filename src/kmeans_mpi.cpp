#include "kmeans_mpi.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void check_mpi(int code, const char* operation) {
    if (code == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " + std::string(message, length));
}

int checked_mpi_count(std::size_t value, const char* description) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(description) + " exceeds MPI's int count limit");
    return static_cast<int>(value);
}
}

KMeansResult run_mpi_kmeans(const Dataset& root_data,
                            const KMeansOptions& options,
                            int rank,
                            int world_size,
                            int thread_count,
                            MPI_Comm communicator) {
    if (world_size <= 0 || rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid MPI rank or world size");
    if (thread_count <= 0)
        throw std::invalid_argument("OpenMP thread count must be positive");

    // The command line is parsed on every rank, so dimensions and K are known
    // consistently without an extra metadata broadcast.
    const std::size_t dimensions = rank == 0 ? root_data.dimensions : 0;
    std::size_t point_count = rank == 0 ? root_data.point_count : 0;
    unsigned long long shape[2] = {
        static_cast<unsigned long long>(point_count),
        static_cast<unsigned long long>(dimensions)
    };
    check_mpi(MPI_Bcast(shape, 2, MPI_UNSIGNED_LONG_LONG, 0, communicator), "MPI_Bcast(dataset shape)");
    point_count = static_cast<std::size_t>(shape[0]);
    const std::size_t dims = static_cast<std::size_t>(shape[1]);
    if (point_count == 0 || dims == 0 || options.clusters == 0 || options.clusters > point_count)
        throw std::invalid_argument("dataset dimensions and cluster count are invalid");
    if (options.max_iterations == 0 || options.tolerance < 0.0 || !std::isfinite(options.tolerance))
        throw std::invalid_argument("maximum iterations must be positive and tolerance non-negative");

    const std::size_t global_value_count = point_count * dims;
    const std::size_t centroid_value_count = options.clusters * dims;
    checked_mpi_count(global_value_count, "dataset element count");
    const int centroid_count = checked_mpi_count(centroid_value_count, "centroid element count");

    std::vector<int> element_counts(static_cast<std::size_t>(world_size));
    std::vector<int> displacements(static_cast<std::size_t>(world_size));
    std::vector<std::size_t> local_point_counts(static_cast<std::size_t>(world_size));
    const std::size_t base_points = point_count / static_cast<std::size_t>(world_size);
    const std::size_t extra_points = point_count % static_cast<std::size_t>(world_size);
    std::size_t point_offset = 0;
    for (int process = 0; process < world_size; ++process) {
        const std::size_t count = base_points + (static_cast<std::size_t>(process) < extra_points ? 1 : 0);
        local_point_counts[static_cast<std::size_t>(process)] = count;
        element_counts[static_cast<std::size_t>(process)] = checked_mpi_count(count * dims, "rank dataset element count");
        displacements[static_cast<std::size_t>(process)] = checked_mpi_count(point_offset * dims, "rank dataset displacement");
        point_offset += count;
    }
    const std::size_t local_point_count = local_point_counts[static_cast<std::size_t>(rank)];
    std::vector<double> local_points(local_point_count * dims);

    // Rank 0 initializes the same seeded K-Means++ centroids as the serial run.
    // Everyone then receives that one agreed-upon initial model.
    std::vector<double> centroids(centroid_value_count);
    int initialization_ok = 1;
    if (rank == 0) {
        try {
            if (root_data.values.size() != global_value_count)
                throw std::invalid_argument("root dataset storage does not match its shape");
            centroids = initialize_kmeans_plus_plus(root_data, options.clusters, options.seed);
        } catch (const std::exception&) {
            initialization_ok = 0;
        }
    }
    check_mpi(MPI_Bcast(&initialization_ok, 1, MPI_INT, 0, communicator), "MPI_Bcast(initialization status)");
    if (!initialization_ok)
        throw std::runtime_error("rank 0 could not validate the dataset or initialize centroids");
    check_mpi(MPI_Bcast(centroids.data(), centroid_count, MPI_DOUBLE, 0, communicator), "MPI_Bcast(centroids)");

    // Scatterv accepts a different element count and offset for each rank. The
    // first remainder ranks receive one extra point, so uneven N is supported.
    const double* send_buffer = rank == 0 ? root_data.values.data() : nullptr;
    check_mpi(MPI_Scatterv(send_buffer,
                           element_counts.data(), displacements.data(), MPI_DOUBLE,
                           local_points.data(), element_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE,
                           0, communicator), "MPI_Scatterv(points)");

    std::vector<double> local_sums(centroid_value_count);
    std::vector<double> global_sums(centroid_value_count);
    std::vector<unsigned long long> local_counts(options.clusters);
    std::vector<unsigned long long> global_counts(options.clusters);
    if (static_cast<std::size_t>(thread_count) >
        std::numeric_limits<std::size_t>::max() / centroid_value_count)
        throw std::invalid_argument("thread-local accumulator size overflows addressable memory");
    const std::size_t thread_sum_size = static_cast<std::size_t>(thread_count) * centroid_value_count;
    const std::size_t thread_count_size = static_cast<std::size_t>(thread_count) * options.clusters;
    std::vector<double> thread_sums(thread_sum_size);
    std::vector<unsigned long long> thread_counts(thread_count_size);
    KMeansResult result;
    result.centroids = centroids;
    result.assignments.resize(local_point_count, -1);

    const double start_time = MPI_Wtime();
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0ULL);
        std::fill(thread_sums.begin(), thread_sums.end(), 0.0);
        std::fill(thread_counts.begin(), thread_counts.end(), 0ULL);
        int iteration_threads = 1;

        // Static scheduling gives each thread a contiguous, predictable share
        // of equal-cost points. Each thread writes only to its own buffers.
        #pragma omp parallel num_threads(thread_count)
        {
            const int thread_id = omp_get_thread_num();
            double* sums = thread_sums.data() + static_cast<std::size_t>(thread_id) * centroid_value_count;
            unsigned long long* counts = thread_counts.data() +
                                         static_cast<std::size_t>(thread_id) * options.clusters;
            #pragma omp single nowait
            iteration_threads = omp_get_num_threads();

            #pragma omp for schedule(static)
            for (std::int64_t point_index = 0;
                 point_index < static_cast<std::int64_t>(local_point_count); ++point_index) {
                const std::size_t i = static_cast<std::size_t>(point_index);
                const double* point = local_points.data() + i * dims;
                std::size_t best_cluster = 0;
                double best_distance = std::numeric_limits<double>::infinity();
                for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
                    const double distance = squared_distance(point, centroids.data() + cluster * dims, dims);
                    if (distance < best_distance) { best_distance = distance; best_cluster = cluster; }
                }
                result.assignments[i] = static_cast<int>(best_cluster);
                ++counts[best_cluster];
                for (std::size_t d = 0; d < dims; ++d)
                    sums[best_cluster * dims + d] += point[d];
            }
        }
        result.threads_used = std::max(result.threads_used, iteration_threads);

        // Merge each thread's private totals into this rank's totals only after
        // the parallel region has joined. MPI then reduces rank totals globally.
        for (int thread_id = 0; thread_id < iteration_threads; ++thread_id) {
            const std::size_t sum_offset = static_cast<std::size_t>(thread_id) * centroid_value_count;
            const std::size_t count_offset = static_cast<std::size_t>(thread_id) * options.clusters;
            for (std::size_t i = 0; i < centroid_value_count; ++i)
                local_sums[i] += thread_sums[sum_offset + i];
            for (std::size_t cluster = 0; cluster < options.clusters; ++cluster)
                local_counts[cluster] += thread_counts[count_offset + cluster];
        }

        // Allreduce sums each rank's partial sums/counts and gives the global
        // totals back to every rank, so every rank computes the same centroids.
        check_mpi(MPI_Allreduce(local_sums.data(), global_sums.data(), centroid_count,
                                MPI_DOUBLE, MPI_SUM, communicator), "MPI_Allreduce(cluster sums)");
        check_mpi(MPI_Allreduce(local_counts.data(), global_counts.data(),
                                checked_mpi_count(options.clusters, "cluster count"),
                                MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator), "MPI_Allreduce(cluster counts)");

        double maximum_shift_squared = 0.0;
        for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
            if (global_counts[cluster] == 0) continue; // Keep the prior center for an empty cluster.
            std::vector<double> updated(dims);
            for (std::size_t d = 0; d < dims; ++d)
                updated[d] = global_sums[cluster * dims + d] /
                             static_cast<double>(global_counts[cluster]);
            maximum_shift_squared = std::max(maximum_shift_squared,
                squared_distance(centroids.data() + cluster * dims, updated.data(), dims));
            std::copy(updated.begin(), updated.end(), centroids.begin() + cluster * dims);
        }
        result.iterations = iteration + 1;
        if (std::sqrt(maximum_shift_squared) <= options.tolerance) {
            result.converged = true; // Same global data means every rank takes this branch.
            break;
        }
    }

    // Measure inertia against the final model; sum local errors on all ranks.
    double local_inertia = 0.0;
    for (std::size_t i = 0; i < local_point_count; ++i) {
        const double* point = local_points.data() + i * dims;
        std::size_t best_cluster = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
            const double distance = squared_distance(point, centroids.data() + cluster * dims, dims);
            if (distance < best_distance) { best_distance = distance; best_cluster = cluster; }
        }
        result.assignments[i] = static_cast<int>(best_cluster);
        local_inertia += best_distance;
    }
    check_mpi(MPI_Allreduce(&local_inertia, &result.inertia, 1, MPI_DOUBLE, MPI_SUM, communicator),
              "MPI_Allreduce(inertia)");

    const double local_elapsed = MPI_Wtime() - start_time;
    check_mpi(MPI_Reduce(&local_elapsed, &result.elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, communicator),
              "MPI_Reduce(runtime)");
    result.centroids = std::move(centroids);
    return result;
}
