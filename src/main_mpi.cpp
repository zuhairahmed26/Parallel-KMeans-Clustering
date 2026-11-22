#include "cli.h"
#include "dataset.h"
#include "kmeans_mpi.h"

#include <omp.h>

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void check_mpi(int code, const char* operation) {
    if (code == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " + std::string(message, length));
}

class MpiSession {
public:
    MpiSession(int& argc, char**& argv) {
        int provided = MPI_THREAD_SINGLE;
        const int status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (status != MPI_SUCCESS) throw std::runtime_error("MPI_Init_thread failed");
        active_ = true;
        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            active_ = false;
            throw std::runtime_error("MPI does not provide the required MPI_THREAD_FUNNELED level");
        }
        const int handler_status = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        if (handler_status != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            active_ = false;
            throw std::runtime_error("MPI_Comm_set_errhandler failed");
        }
    }

    ~MpiSession() {
        if (!active_) return;
        // Fallback protection if an exception escapes the coordinated handler.
        if (std::uncaught_exceptions() > 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
        MPI_Finalize();
    }

    MpiSession(const MpiSession&) = delete;
    MpiSession& operator=(const MpiSession&) = delete;

private:
    bool active_ = false;
};

int run_mpi_job(int argc, char** argv, int rank, int world_size) {
    CommandLineOptions cli;
    bool help = false;
    std::string error;
    const bool parsed = parse_command_line(argc, argv, cli, help, error);
    int all_parsed = parsed ? 1 : 0;
    check_mpi(MPI_Allreduce(MPI_IN_PLACE, &all_parsed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(argument status)");
    if (!all_parsed) {
        if (rank == 0) std::cerr << "Error: " << (parsed ? "invalid arguments on an MPI rank" : error)
                                 << "\nUse --help for usage.\n";
        return 2;
    }
    if (help) {
        if (rank == 0) print_usage(argv[0]);
        return 0;
    }

    // MPI_Scatterv uses int counts/displacements. Reject unrepresentable
    // inputs early and make every process take the same exit path.
    int shape_ok = 1;
    if (cli.points > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                     std::max<std::size_t>(cli.dimensions, 1)) shape_ok = 0;
    if (cli.kmeans.clusters > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                              std::max<std::size_t>(cli.dimensions, 1)) shape_ok = 0;
    check_mpi(MPI_Allreduce(MPI_IN_PLACE, &shape_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(shape status)");
    if (!shape_ok) {
        if (rank == 0) std::cerr << "Error: dataset or centroid size exceeds MPI's int count limit\n";
        return 2;
    }

    // CLI overrides OMP_NUM_THREADS. Without --threads, use the OpenMP
    // runtime's environment/default value. Disable dynamic shrinking so
    // the selected team size can be observed and reproduced.
    omp_set_dynamic(0);
    const int thread_count = cli.threads == 0
        ? omp_get_max_threads()
        : static_cast<int>(cli.threads);
    omp_set_num_threads(thread_count);

    Dataset root_data;
    int generation_ok = 1;
    std::string generation_error;
    if (rank == 0) {
        try {
            root_data = generate_gaussian_blobs(cli.points, cli.dimensions,
                cli.kmeans.clusters, cli.kmeans.seed, cli.standard_deviation);
        } catch (const std::exception& exception) {
            generation_ok = 0;
            generation_error = exception.what();
        }
    }
    check_mpi(MPI_Bcast(&generation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD),
              "MPI_Bcast(dataset generation status)");
    if (!generation_ok) {
        if (rank == 0) std::cerr << "Error: " << generation_error << '\n';
        return 1;
    }

    const KMeansResult result = run_mpi_kmeans(root_data, cli.kmeans, rank, world_size, thread_count);
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(12)
                  << "Mode: hybrid MPI + OpenMP\nPoints: " << cli.points
                  << "\nDimensions: " << cli.dimensions << "\nClusters: " << cli.kmeans.clusters
                  << "\nMPI ranks: " << world_size << "\nOpenMP threads per rank: " << result.threads_used
                  << "\nIterations: " << result.iterations
                  << "\nConverged: " << (result.converged ? "yes" : "no")
                  << "\nRuntime (seconds): " << result.elapsed_seconds
                  << "\nInertia (SSE): " << result.inertia << "\nCentroids:\n";
        for (std::size_t c = 0; c < cli.kmeans.clusters; ++c) {
            std::cout << "  " << c << ": [";
            for (std::size_t d = 0; d < cli.dimensions; ++d) {
                if (d) std::cout << ", ";
                std::cout << result.centroids[c * cli.dimensions + d];
            }
            std::cout << "]\n";
        }
    }
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        MpiSession mpi(argc, argv);
        int rank = 0;
        int world_size = 0;
        try {
            check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
            check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
            return run_mpi_job(argc, argv, rank, world_size);
        } catch (const std::exception& exception) {
            std::cerr << "MPI error on rank " << rank << ": " << exception.what() << '\n';
            // Keep MPI initialized while aborting, so peers cannot remain
            // stranded in a collective if this process fails locally.
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    } catch (const std::exception& exception) {
        std::cerr << "MPI startup error: " << exception.what() << '\n';
        return 1;
    }
}
