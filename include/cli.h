#pragma once

#include "kmeans.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct CommandLineOptions {
    std::size_t points = 10000;
    std::size_t dimensions = 2;
    // Zero means: use the OpenMP runtime setting (for example OMP_NUM_THREADS).
    std::size_t threads = 0;
    KMeansOptions kmeans;
    double standard_deviation = 1.0;
};

bool parse_command_line(int argc, char** argv, CommandLineOptions& options,
                        bool& show_help, std::string& error);
void print_usage(const char* program_name);
