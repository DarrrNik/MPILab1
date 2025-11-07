#include <mpi.h>
#include <fstream>
#include <sstream>
#include <format>
#include <iostream>
#include <cassert>
#include <cstdlib>

#include "mpilab.hpp"

int ParseCLArguments(int argc, char** argv, int commRank, int* rows, int* cols, char** outputFile)
{
    constexpr int EXPECTED_ARGS = 3; //"имя программы", "строки", "столбцы"
    constexpr int ARG_ROWS = 1;
    constexpr int ARG_COLS = 2;
    constexpr int ARG_OUTPUT_FILE = 3;

    if (argc < EXPECTED_ARGS)
    {
        if (commRank == ROOT_PROCESS)
        {
            std::cerr << "Error: missing arguments <rows> <cols>" << std::endl;
        }
        return 1;
    }

    *rows = std::atoi(argv[ARG_ROWS]);
    if (*rows <= 0 || std::string(argv[ARG_ROWS]) != std::to_string(*rows))
    {
        if (commRank == ROOT_PROCESS)
        {
            std::cerr << "Error: <rows> must be positive integer" << std::endl;
        }
        return 2;
    }


    *cols = atoi(argv[ARG_COLS]);
    if (*cols <= 0 || std::string(argv[ARG_COLS]) != std::to_string(*cols))
    {
        if (commRank == ROOT_PROCESS)
        {
            std::cerr << "Error: <cols> must be positive integer" << std::endl;
        }
        return 3;
    }

    *outputFile = (argc > EXPECTED_ARGS) ? argv[ARG_OUTPUT_FILE] : nullptr;
    return 0;
}
