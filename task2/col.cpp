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
        vector = new double[cols];
        result = new double[rows];
        FillMatrixVectorCols(rows, cols, matrix, vector);
    }

    double* localMatrix = nullptr, * localVector = nullptr, * localResult = nullptr;
    int localCols = DistributeMatrixCols(rows, cols, matrix, vector, commRank, commSize, &localMatrix, &localVector);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MultiplyMatrixVectorCols(rows, localCols, localMatrix, localVector, &localResult);
    MPI_Reduce(localResult, result, rows, MPI_DOUBLE, MPI_SUM, ROOT_PROCESS, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double localTime = end - start, globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT_PROCESS, MPI_COMM_WORLD);

    if (commRank == ROOT_PROCESS) 
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Matrix-Vector multiplication (column-wise, MPI)"  << std::endl;
        std::cout << "> Processes         <=> " << commSize             << std::endl;
        std::cout << "> Matrix size       <=> " << rows << " x " << cols << std::endl;
        std::cout << "> Execution time    <=> " << globalTime << 's'    << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        std::fstream file{ outputFile, std::ios::ate };
        if (file) 
        {
            std::stringstream methodName;
            methodName << "mul_cols_" << rows << '_' << cols;
            file << std::format("{},{},{}\n", methodName.str(), commSize, globalTime);
        }
    }

    delete[] localMatrix; delete[] localVector; delete[] localResult;
    if (commRank == ROOT_PROCESS) 
    { 
        delete[] vector; delete[] matrix; delete[] result; 
    }

    MPI_Finalize();
}

int DistributeMatrixCols(int rows, int cols, double* matrix, double* vector, int commRank, int commSize, double** localMatrix, double** localVector) 
{
    int* colCounts = nullptr, * colDispls = nullptr;
    ComputeCountsDispls(cols, commSize, &colCounts, &colDispls);
    int localCols = colCounts[commRank];

    *localMatrix = new double[rows * localCols];
    *localVector = new double[localCols];

    if (commRank == ROOT_PROCESS) 
    {
        for (int i = 0; i < commSize; i++) 
        {
            int startCol = colDispls[i];
            int sendCols = colCounts[i];
            if (i == ROOT_PROCESS) 
            {
                for (int c = 0; c < sendCols; c++)
                    for (int r = 0; r < rows; r++)
                        (*localMatrix)[r * sendCols + c] = matrix[r * cols + (startCol + c)];
            }
            else 
            {
                MPI_Datatype MPI_COL_BLOCK;
                MPI_Type_vector(rows, sendCols, cols, MPI_DOUBLE, &MPI_COL_BLOCK);
                MPI_Type_commit(&MPI_COL_BLOCK);

                MPI_Send(&matrix[startCol], 1, MPI_COL_BLOCK, i, 0, MPI_COMM_WORLD);
                MPI_Type_free(&MPI_COL_BLOCK);
            }
        }
    }
    else 
    {
        MPI_Recv(*localMatrix, rows * localCols, MPI_DOUBLE, ROOT_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Scatterv(vector, colCounts, colDispls, MPI_DOUBLE, *localVector, localCols, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);

    delete[] colCounts; delete[] colDispls;
    return localCols;
}

void MultiplyMatrixVectorCols(int rows, int localCols, double* localMatrix, double* localVector, double** localResult) 
{
    *localResult = new double[rows];
    for (int c = 0; c < localCols; c++) 
    {
        for (int r = 0; r < rows; r++) 
        {
            (*localResult)[r] += localMatrix[r * localCols + c] * localVector[c];
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

void FillMatrixVectorCols(int rows, int cols, double* matrix, double* vector)
{
    std::srand(static_cast<int>(std::time({})));
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = static_cast<double>(std::rand());
    }
    for (int i = 0; i < cols; i++)
    {
        vector[i] = static_cast<double>(std::rand());
    }
}
