#include "cli.h"
#include "dataset.h"
#include "kmeans_serial.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    CommandLineOptions cli;
    bool help = false;
    std::string error;
    if (!parse_command_line(argc, argv, cli, help, error)) {
        std::cerr << "Error: " << error << "\nUse --help for usage.\n";
        return 2;
    }
    if (help) { print_usage(argv[0]); return 0; }

    try {
        const Dataset data = generate_gaussian_blobs(cli.points, cli.dimensions,
            cli.kmeans.clusters, cli.kmeans.seed, cli.standard_deviation);
        const auto initial = initialize_kmeans_plus_plus(data, cli.kmeans.clusters, cli.kmeans.seed);
        const KMeansResult result = run_serial_kmeans(data, cli.kmeans, initial);

        std::cout << std::fixed << std::setprecision(6)
                  << "Mode: serial\nPoints: " << cli.points << "\nDimensions: " << cli.dimensions
                  << "\nClusters: " << cli.kmeans.clusters << "\nIterations: " << result.iterations
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
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
