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

    if (commSize > 1 && cols % commSize != 0) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Error: cols must be divisible by number of processes" << std::endl;
            std::cerr << "Cols: " << cols << ", Processes: " << commSize << std::endl;
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
            
            for (int i = 0; i < rows; i++) {
                result[i] = 0.0;
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = 0.0;
        }
    }

    double* localMatrix = nullptr;
    double* localVector = nullptr;
    double* localResult = nullptr;
    
    int localCols = DistributeMatrixCols(rows, cols, matrix, vector, commRank, commSize, &localMatrix, &localVector);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MultiplyMatrixVectorCols(rows, localCols, localMatrix, localVector, &localResult);
    
    MPI_Reduce(localResult, result, rows, MPI_DOUBLE, MPI_SUM, ROOT_PROCESS, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double localTime = end - start;
    double globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT_PROCESS, MPI_COMM_WORLD);

    if (commRank == ROOT_PROCESS) 
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Matrix-Vector multiplication (column-wise, MPI)"  << std::endl;
        std::cout << "> Processes         <=> " << commSize             << std::endl;
        std::cout << "> Matrix size       <=> " << rows << " x " << cols << std::endl;
        std::cout << "> Execution time    <=> " << globalTime << 's'    << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    delete[] localMatrix; 
    delete[] localVector; 
    delete[] localResult;
    
    if (commRank == ROOT_PROCESS) 
    { 
        delete[] matrix; 
        delete[] vector; 
        delete[] result; 
    } else {
        delete[] result;
    }

    MPI_Finalize();
    return 0;
}

int DistributeMatrixCols(int rows, int cols, double* matrix, double* vector, int commRank, int commSize, double** localMatrix, double** localVector) 
{
    int* colCounts = nullptr, * colDispls = nullptr;
    ComputeCountsDispls(cols, commSize, &colCounts, &colDispls);
    int localCols = colCounts[commRank];

    *localMatrix = new double[rows * localCols];
    *localVector = new double[localCols];
    
    for (int i = 0; i < rows * localCols; i++) {
        (*localMatrix)[i] = 0.0;
    }
    
    for (int i = 0; i < localCols; i++) {
        (*localVector)[i] = 0.0;
    }

    if (commRank == ROOT_PROCESS) 
    {
        for (int i = 0; i < commSize; i++) 
        {
            int startCol = colDispls[i];
            int sendCols = colCounts[i];
            
            if (i == ROOT_PROCESS) 
            {
                for (int c = 0; c < sendCols; c++) {
                    for (int r = 0; r < rows; r++) {
                        (*localMatrix)[r * sendCols + c] = matrix[r * cols + (startCol + c)];
                    }
                }
            }
            else 
            {
                double* sendBuffer = new double[rows * sendCols];
                for (int c = 0; c < sendCols; c++) {
                    for (int r = 0; r < rows; r++) {
                        sendBuffer[r * sendCols + c] = matrix[r * cols + (startCol + c)];
                    }
                }
                
                MPI_Send(sendBuffer, rows * sendCols, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                delete[] sendBuffer;
            }
        }
    }
    else 
    {
        MPI_Recv(*localMatrix, rows * localCols, MPI_DOUBLE, ROOT_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Scatterv(vector, colCounts, colDispls, MPI_DOUBLE, 
                 *localVector, localCols, MPI_DOUBLE, 
                 ROOT_PROCESS, MPI_COMM_WORLD);

    delete[] colCounts; 
    delete[] colDispls;
    return localCols;
}

void MultiplyMatrixVectorCols(int rows, int localCols, double* localMatrix, double* localVector, double** localResult) 
{
    *localResult = new double[rows];
    
    for (int r = 0; r < rows; r++) {
        (*localResult)[r] = 0.0;
    }
    
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