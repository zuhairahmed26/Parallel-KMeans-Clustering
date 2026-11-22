#include "cli.h"
#include "dataset.h"
#include "kmeans_mpi.h"

#include <exception>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class MpiSession {
public:
    MpiSession(int& argc, char**& argv) {
        const int status = MPI_Init(&argc, &argv);
        if (status != MPI_SUCCESS) throw std::runtime_error("MPI_Init failed");
        active_ = true;
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    }
    ~MpiSession() { if (active_) MPI_Finalize(); }
    MpiSession(const MpiSession&) = delete;
    MpiSession& operator=(const MpiSession&) = delete;
private:
    bool active_ = false;
};
}

int main(int argc, char** argv) {
    try {
        MpiSession mpi(argc, argv);
        int rank = 0;
        int world_size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        CommandLineOptions cli;
        bool help = false;
        std::string error;
        const bool parsed = parse_command_line(argc, argv, cli, help, error);
        int all_parsed = parsed ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &all_parsed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
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
        MPI_Allreduce(MPI_IN_PLACE, &shape_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!shape_ok) {
            if (rank == 0) std::cerr << "Error: dataset or centroid size exceeds MPI's int count limit\n";
            return 2;
        }

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
        MPI_Bcast(&generation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!generation_ok) {
            if (rank == 0) std::cerr << "Error: " << generation_error << '\n';
            return 1;
        }

        const KMeansResult result = run_mpi_kmeans(root_data, cli.kmeans, rank, world_size);
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(12)
                      << "Mode: MPI (serial assignment per rank)\nPoints: " << cli.points
                      << "\nDimensions: " << cli.dimensions << "\nClusters: " << cli.kmeans.clusters
                      << "\nMPI ranks: " << world_size << "\nOpenMP threads per rank: 1"
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
    } catch (const std::exception& exception) {
        std::cerr << "MPI error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
