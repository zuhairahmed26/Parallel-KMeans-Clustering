#include "cli.h"

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
std::size_t parse_size(const char* text, const char* name) {
    if (text[0] == '\0' || text[0] == '-')
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value == 0 || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return static_cast<std::size_t>(value);
}
std::uint64_t parse_seed(const char* text) {
    if (text[0] == '\0' || text[0] == '-')
        throw std::invalid_argument("seed must be a non-negative integer");
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value > std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("seed must be a non-negative integer in range");
    return static_cast<std::uint64_t>(value);
}
double parse_double(const char* text, const char* name, bool allow_zero) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (text[0] == '\0' || *end != '\0' || !std::isfinite(value) || value < 0.0 || (!allow_zero && value == 0.0))
        throw std::invalid_argument(std::string(name) + " must be a finite " + (allow_zero ? "non-negative" : "positive") + " number");
    return value;
}
}

bool parse_command_line(int argc, char** argv, CommandLineOptions& options,
                        bool& show_help, std::string& error) {
    show_help = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { show_help = true; return true; }
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + arg);
            const char* value = argv[++i];
            if (arg == "--points") options.points = parse_size(value, "points");
            else if (arg == "--dimensions") options.dimensions = parse_size(value, "dimensions");
            else if (arg == "--threads") {
                options.threads = parse_size(value, "threads");
                if (options.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("threads exceeds the supported OpenMP thread count");
            }
            else if (arg == "--clusters") options.kmeans.clusters = parse_size(value, "clusters");
            else if (arg == "--max-iterations") options.kmeans.max_iterations = parse_size(value, "max iterations");
            else if (arg == "--seed") options.kmeans.seed = parse_seed(value);
            else if (arg == "--tolerance") options.kmeans.tolerance = parse_double(value, "tolerance", true);
            else if (arg == "--stddev") options.standard_deviation = parse_double(value, "stddev", false);
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (options.kmeans.clusters > options.points)
            throw std::invalid_argument("clusters cannot exceed points");
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --points N           Number of synthetic points (default 10000)\n"
              << "  --dimensions D       Coordinates per point (default 2)\n"
              << "  --threads N          OpenMP threads per MPI rank (MPI executable only)\n"
              << "  --clusters K         Number of clusters (default 3)\n"
              << "  --max-iterations N   Iteration limit (default 100)\n"
              << "  --tolerance X        Centroid movement threshold (default 0.0001)\n"
              << "  --seed N             Reproducible random seed (default 42)\n"
              << "  --stddev X           Gaussian blob standard deviation (default 1)\n"
              << "  --help               Show this help\n";
}
