#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

#include "mpilab.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int commRank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    // Проверяем аргументы командной строки
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

    // Для блочного распределения нужна квадратная сетка процессов
    int gridSize = static_cast<int>(std::sqrt(commSize));
    if (gridSize * gridSize != commSize) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Error: number of processes must be a perfect square for block distribution" << std::endl;
            std::cerr << "Processes: " << commSize << ", Perfect squares: 1, 4, 9, 16, ..." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Проверяем делимость на gridSize
    if (rows % gridSize != 0 || cols % gridSize != 0) {
        if (commRank == ROOT_PROCESS) {
            std::cerr << "Error: rows and cols must be divisible by sqrt(processes)" << std::endl;
            std::cerr << "Rows: " << rows << ", Cols: " << cols << ", Grid size: " << gridSize << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Выделяем память для ROOT процесса
    double* matrix = nullptr;
    double* vector = nullptr; 
    double* result = nullptr;
    
    if (commRank == ROOT_PROCESS) {
        try {
            matrix = new double[rows * cols];
            vector = new double[cols];
            result = new double[rows];
            
            // Инициализируем данные
            FillMatrixVectorBlocks(rows, cols, matrix, vector);
            
            // Инициализируем результат нулями
            for (int i = 0; i < rows; i++) {
                result[i] = 0.0;
            }
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        // Не-ROOT процессы выделяют память только для результата
        result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = 0.0;
        }
    }

    // Распределяем данные
    double* localMatrix = nullptr;
    double* localVector = nullptr;
    double* localResult = nullptr;
    int localRows, localCols;
    
    DistributeMatrixBlocks(rows, cols, matrix, vector, commRank, commSize, 
                          &localMatrix, &localVector, &localRows, &localCols);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Умножаем локальную часть
    MultiplyMatrixVectorBlocks(localRows, localCols, localMatrix, localVector, &localResult);
    
    // Собираем результаты
    GatherResultsBlocks(localResult, result, commRank, commSize, rows, cols, localRows, localCols);

    double end = MPI_Wtime();
    double localTime = end - start;
    double globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT_PROCESS, MPI_COMM_WORLD);

    if (commRank == ROOT_PROCESS)
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Matrix-Vector multiplication (block-wise, MPI)" << std::endl;
        std::cout << "> Processes         <=> " << commSize << std::endl;
        std::cout << "> Matrix size       <=> " << rows << " x " << cols << std::endl;
        std::cout << "> Grid size         <=> " << gridSize << " x " << gridSize << std::endl;
        std::cout << "> Execution time    <=> " << globalTime << 's' << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    // Освобождаем память
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

void FillMatrixVectorBlocks(int rows, int cols, double* matrix, double* vector)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<double>(std::rand() % 100) / 10.0;
    }
    for (int i = 0; i < cols; i++) {
        vector[i] = static_cast<double>(std::rand() % 100) / 10.0;
    }
}

void DistributeMatrixBlocks(int rows, int cols, double* matrix, double* vector, 
                           int commRank, int commSize, 
                           double** localMatrix, double** localVector,
                           int* localRows, int* localCols)
{
    int gridSize = static_cast<int>(std::sqrt(commSize));
    
    // Вычисляем координаты процесса в сетке
    int rowCoord = commRank / gridSize;
    int colCoord = commRank % gridSize;
    
    // Вычисляем локальные размеры
    *localRows = rows / gridSize;
    *localCols = cols / gridSize;
    
    // Выделяем память для локальных данных
    *localMatrix = new double[(*localRows) * (*localCols)];
    *localVector = new double[*localCols];
    
    // Инициализируем нулями
    for (int i = 0; i < (*localRows) * (*localCols); i++) {
        (*localMatrix)[i] = 0.0;
    }
    for (int i = 0; i < *localCols; i++) {
        (*localVector)[i] = 0.0;
    }

    if (commRank == ROOT_PROCESS) {
        // ROOT процесс распределяет данные
        
        // Сначала распределяем вектор (каждому процессу в столбце нужна своя часть вектора)
        for (int proc = 0; proc < commSize; proc++) {
            int procRow = proc / gridSize;
            int procCol = proc % gridSize;
            int startCol = procCol * (*localCols);
            
            if (proc == ROOT_PROCESS) {
                // Копируем часть вектора для ROOT процесса
                for (int c = 0; c < *localCols; c++) {
                    (*localVector)[c] = vector[startCol + c];
                }
            } else {
                // Отправляем часть вектора другим процессам
                MPI_Send(&vector[startCol], *localCols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }
        }
        
        // Затем распределяем матрицу по блокам
        for (int proc = 0; proc < commSize; proc++) {
            int procRow = proc / gridSize;
            int procCol = proc % gridSize;
            int startRow = procRow * (*localRows);
            int startCol = procCol * (*localCols);
            
            if (proc == ROOT_PROCESS) {
                // Копируем блок матрицы для ROOT процесса
                for (int r = 0; r < *localRows; r++) {
                    for (int c = 0; c < *localCols; c++) {
                        (*localMatrix)[r * (*localCols) + c] = 
                            matrix[(startRow + r) * cols + (startCol + c)];
                    }
                }
            } else {
                // Отправляем блок матрицы другим процессам
                double* sendBuffer = new double[(*localRows) * (*localCols)];
                for (int r = 0; r < *localRows; r++) {
                    for (int c = 0; c < *localCols; c++) {
                        sendBuffer[r * (*localCols) + c] = 
                            matrix[(startRow + r) * cols + (startCol + c)];
                    }
                }
                MPI_Send(sendBuffer, (*localRows) * (*localCols), MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                delete[] sendBuffer;
            }
        }
    } else {
        // Не-ROOT процессы получают данные
        
        // Получаем часть вектора
        MPI_Recv(*localVector, *localCols, MPI_DOUBLE, ROOT_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Получаем блок матрицы
        MPI_Recv(*localMatrix, (*localRows) * (*localCols), MPI_DOUBLE, ROOT_PROCESS, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void MultiplyMatrixVectorBlocks(int localRows, int localCols, 
                               double* localMatrix, double* localVector, 
                               double** localResult)
{
    *localResult = new double[localRows];
    
    // Инициализируем результат нулями
    for (int r = 0; r < localRows; r++) {
        (*localResult)[r] = 0.0;
    }
    
    // Умножаем локальный блок матрицы на локальную часть вектора
    for (int r = 0; r < localRows; r++) {
        for (int c = 0; c < localCols; c++) {
            (*localResult)[r] += localMatrix[r * localCols + c] * localVector[c];
        }
    }
}

void GatherResultsBlocks(double* localResult, double* result,
                        int commRank, int commSize,
                        int rows, int cols, int localRows, int localCols)
{
    int gridSize = static_cast<int>(std::sqrt(commSize));
    
    if (commRank == ROOT_PROCESS) {
        // ROOT процесс собирает результаты
        
        // Сначала копируем свой локальный результат
        int rowCoord = commRank / gridSize;
        int colCoord = commRank % gridSize;
        int startRow = rowCoord * localRows;
        
        for (int r = 0; r < localRows; r++) {
            result[startRow + r] = localResult[r];
        }
        
        // Затем получаем результаты от других процессов
        for (int proc = 1; proc < commSize; proc++) {
            int procRow = proc / gridSize;
            int procCol = proc % gridSize;
            int procStartRow = procRow * localRows;
            
            double* recvBuffer = new double[localRows];
            MPI_Recv(recvBuffer, localRows, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int r = 0; r < localRows; r++) {
                result[procStartRow + r] = recvBuffer[r];
            }
            
            delete[] recvBuffer;
        }
    } else {
        // Не-ROOT процессы отправляют свои результаты ROOT процессу
        MPI_Send(localResult, localRows, MPI_DOUBLE, ROOT_PROCESS, 2, MPI_COMM_WORLD);
    }
}