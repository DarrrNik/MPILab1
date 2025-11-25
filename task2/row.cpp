#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "mpilab.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int commRank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if (argc < 3) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);
    
    if (rows <= 0 || cols <= 0) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Error: rows and cols must be positive integers" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (commSize > 1 && rows % commSize != 0) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Error: rows must be divisible by number of processes" << std::endl;
            std::cerr << "Rows: " << rows << ", Processes: " << commSize << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    double* matrix = nullptr;
    double* vector = nullptr; 
    double* result = nullptr;
    
    if (commRank == ROOT_PROCESS) {
        try {
            matrix = new double[rows * cols];
            vector = new double[cols];
            result = new double[rows];
            
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
            for (int i = 0; i < rows * cols; i++) {
                matrix[i] = static_cast<double>(std::rand() % 100) / 10.0;
            }
            for (int i = 0; i < cols; i++) {
                vector[i] = static_cast<double>(std::rand() % 100) / 10.0;
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double* localVector = new double[cols];
    
    if (commRank == ROOT_PROCESS) {
        std::copy(vector, vector + cols, localVector);
    }

    MPI_Bcast(localVector, cols, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);

    int localRows = rows / commSize;
    double* localMatrix = new double[localRows * cols];
    double* localResult = new double[localRows];

    MPI_Scatter(matrix, localRows * cols, MPI_DOUBLE,
                localMatrix, localRows * cols, MPI_DOUBLE,
                ROOT_PROCESS, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int r = 0; r < localRows; ++r) {
        double sum = 0.0;
        int rowOffset = r * cols;
        for (int c = 0; c < cols; ++c) {
            sum += localMatrix[rowOffset + c] * localVector[c];
        }
        localResult[r] = sum;
    }

    MPI_Gather(localResult, localRows, MPI_DOUBLE,
               result, localRows, MPI_DOUBLE,
               ROOT_PROCESS, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double localTime = end - start;
    double globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT_PROCESS, MPI_COMM_WORLD);

    if (commRank == ROOT_PROCESS) {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Matrix-Vector multiplication (row-wise, MPI)" << std::endl;
        std::cout << "> Processes         <=> " << commSize << std::endl;
        std::cout << "> Matrix size       <=> " << rows << " x " << cols << std::endl;
        std::cout << "> Execution time    <=> " << globalTime << 's' << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    delete[] localMatrix;
    delete[] localResult;
    delete[] localVector;
    
    if (commRank == ROOT_PROCESS) {
        delete[] matrix;
        delete[] vector;
        delete[] result;
    }

    MPI_Finalize();
    return 0;
}