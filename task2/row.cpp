#include <mpi.h>
#include <fstream>
#include <sstream>
#include <format>
#include <iostream>
#include <cassert>
#include <cstdlib>

#include "mpilab.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int commRank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    int rows = 0, cols = 0; char* outputFile = nullptr;
    {
        int parseRes = ParseCLArguments(argc, argv, commRank, &rows, &cols, &outputFile);
        if (parseRes != 0)
        {
            std::cout << "Invalid command line arguments, check error output" << std::endl;
            return parseRes;
        }
    }

    double* matrix = nullptr, * vector = nullptr, * result = nullptr;
    if (commRank == ROOT_PROCESS)
    {
        matrix = new double[rows * cols];
        vector = new double[rows];
        result = new double[cols];
        FillMatrixVectorRows(rows, cols, matrix, vector);
    }

    double* localMatrix = nullptr, * localResult = nullptr;
    int localCols = DistributeMatrixRows(rows, cols, matrix, vector, commRank, commSize, &localMatrix);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MultiplyMatrixVectorRows(rows, localCols, localMatrix, vector, &localResult);
    MPI_Reduce(localResult, result, rows, MPI_DOUBLE, MPI_SUM, ROOT_PROCESS, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double localTime = end - start, globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT_PROCESS, MPI_COMM_WORLD);

    if (commRank == ROOT_PROCESS)
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Matrix-Vector multiplication (row-wise, MPI)" << std::endl;
        std::cout << "> Processes         <=> " << commSize << std::endl;
        std::cout << "> Matrix size       <=> " << rows << " x " << cols << std::endl;
        std::cout << "> Execution time    <=> " << globalTime << 's' << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        std::fstream file{ outputFile, std::ios::ate };
        if (file)
        {
            std::stringstream methodName;
            methodName << "mul_cols_" << rows << '_' << cols;
            file << std::format("{},{},{}\n", methodName.str(), commSize, globalTime);
        }
    }

    delete[] localMatrix; delete[] localResult;
    if (commRank == ROOT_PROCESS)
    {
        delete[] vector; delete[] matrix; delete[] result;
    }

    MPI_Finalize();
}

int DistributeMatrixRows(int rows, int cols, double* matrix, double* vector, int commRank, int commSize, double** localMatrix)
{
    int* sendCounts = nullptr, * displs = nullptr;
    ComputeCountsDispls(rows, commSize, &sendCounts, &displs);

    int localRows = sendCounts[commRank];
    *localMatrix = new double[localRows * cols];

    MPI_Scatterv(matrix, sendCounts, displs, MPI_DOUBLE, *localMatrix, localRows, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);

    delete[] sendCounts; delete[] displs;
    return localRows;
}

void MultiplyMatrixVectorRows(int localRows, int cols, double* localMatrix, double* vector, double** localResult)
{
    *localResult = new double[localRows];
    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < cols; j++) {
            (*localResult)[i] += localMatrix[i * cols + j] * vector[j];
        }
    }
}

void ComputeCountsDispls(int totalCols, int commSize, int** counts, int** displs)
{
    int base = totalCols / commSize;
    int rem = totalCols % commSize;
    int offset = 0;

    *counts = new int[commSize];
    *displs = new int[commSize];

    for (int i = 0; i < commSize; i++)
    {
        (*counts)[i] = base + (i < rem ? 1 : 0);
        (*displs)[i] = offset;
        offset += (*counts)[i];
    }
}

void FillMatrixVectorRows(int rows, int cols, double* matrix, double* vector)
{
    std::srand(static_cast<int>(std::time({})));
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = static_cast<double>(std::rand());
    }
    for (int i = 0; i < rows; i++)
    {
        vector[i] = static_cast<double>(std::rand());
    }
}
