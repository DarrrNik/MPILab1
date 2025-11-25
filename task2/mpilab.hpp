#ifndef MPILAB_HPP
#define MPILAB_HPP

constexpr int ROOT_PROCESS = 0;

/**
* @brief Парсинг аргументов командной строки
*
* @param[in]    argc        Количество аргументов командной строки
* @param[in]    argv        Массив аргументов командной строки
* @param[in]    commRank    Идентификатор текущего процесса
* @param[out]   rows        Количество строк в матрице
* @param[out]   cols        Количество столбцов в матрице
* @param[out]   outputFile  Путь к файлу с резулитатами работы
*
* @return       0           Парсинг аргументов прошел успешно
* @return       1           Не введены аргументы для создания матрицы
* @return       2           Введено отрицательное число для количества строк
* @return       3           Введено отрицательное число для количества столбцов
*/
int ParseCLArguments(int argc, char** argv, int commRank, int* rows, int* cols, char** outputFile);

/**
* @brief Заполнение матрицы и вектора случайными элементами
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[out]	matrix		Заполненная матрица
* @param[out]	vector		Заполненный вектор
* 
* @note Количество элементов в векторе равно количеству элементов в строке матрицы, другими словами вектор-строка
*/
void FillMatrixVectorCols(int rows, int cols, double* matrix, double* vector);

/**
* @brief Распределение столбцов по процессам
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[in]	matrix		Заполненная матрица
* @param[in]	vector		Заполненный вектор 
* @param[in]    commRank    Идентификатор текущего процесса
* @param[in]    commSize    Общее количество процессов
* @param[out]	localMatrix Матрица текущего процесса
* @param[out]	localVector Вектор текущего процесса
*
* @return Количество столбцов в матрице текущего процесса
*/
int DistributeMatrixCols(int rows, int cols, double* matrix, double* vector, int commRank, int commSize, double** localMatrix, double** localVector);

/**
* @brief Перемножение матрицы на вектор (колонками)
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	localCols   Количество столбцов в матрице
* @param[in]	localMatrix Матрица текущего процесса
* @param[in]	localVector Вектор текущего процесса
* @param[out]	localResult Результирующая матрица
*/
void MultiplyMatrixVectorCols(int rows, int localCols, double* localMatrix, double* localVector, double** localResult);


/**
* @brief Заполнение матрицы и вектора случайными элементами
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[out]	matrix		Заполненная матрица
* @param[out]	vector		Заполненный вектор
*
* @note Количество элементов в векторе равно количеству элементов в строке матрицы, другими словами вектор-столбец
*/
void FillMatrixVectorRows(int rows, int cols, double* matrix, double* vector);

/**
* @brief Распределение строк по процессам
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[in]	matrix		Заполненная матрица
* @param[in]	vector		Заполненный вектор
* @param[in]    commRank    Идентификатор текущего процесса
* @param[in]    commSize    Общее количество процессов
* @param[out]	localMatrix Матрица текущего процесса
* @param[out]	localVector Вектор текущего процесса
*
* @return Количество строк в матрице текущего процесса
*/
int DistributeMatrixRows(int rows, int cols, double* matrix, double* vector, int commRank, int commSize, double** localMatrix);

/**
* @brief Перемножение матрицы на вектор (строками)
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	localCols   Количество столбцов в матрице
* @param[in]	localMatrix Матрица текущего процесса
* @param[in]	localVector Вектор текущего процесса
* @param[out]	localResult Результирующая матрица
*/
void MultiplyMatrixVectorRows(int rows, int localCols, double* localMatrix, double* localVector, double** localResult);

/**
* @brief Заполнение матрицы и вектора случайными элементами (для блочного метода)
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[out]	matrix		Заполненная матрица
* @param[out]	vector		Заполненный вектор
*/
void FillMatrixVectorBlocks(int rows, int cols, double* matrix, double* vector);

/**
* @brief Распределение блоков по процессам
*
* @param[in]	rows        Количество строк в матрице
* @param[in]	cols        Количество столбцов в матрице
* @param[in]	matrix		Заполненная матрица
* @param[in]	vector		Заполненный вектор
* @param[in]    commRank    Идентификатор текущего процесса
* @param[in]    commSize    Общее количество процессов
* @param[out]	localMatrix Матрица текущего процесса
* @param[out]	localVector Вектор текущего процесса
* @param[out]	localRows   Количество строк в блоке текущего процесса
* @param[out]	localCols   Количество столбцов в блоке текущего процесса
*/
void DistributeMatrixBlocks(int rows, int cols, double* matrix, double* vector, 
                           int commRank, int commSize, 
                           double** localMatrix, double** localVector,
                           int* localRows, int* localCols);

/**
* @brief Перемножение матрицы на вектор (блоками)
*
* @param[in]	localRows   Количество строк в блоке текущего процесса
* @param[in]	localCols   Количество столбцов в блоке текущего процесса
* @param[in]	localMatrix Матрица текущего процесса
* @param[in]	localVector Вектор текущего процесса
* @param[out]	localResult Результирующая матрица
*/
void MultiplyMatrixVectorBlocks(int localRows, int localCols, 
                               double* localMatrix, double* localVector, 
                               double** localResult);

/**
* @brief Сбор результатов от всех процессов (для блочного метода)
*
* @param[in]	localResult Локальный результат текущего процесса
* @param[out]	result      Итоговый результат (только для ROOT процесса)
* @param[in]    commRank    Идентификатор текущего процесса
* @param[in]    commSize    Общее количество процессов
* @param[in]    rows        Общее количество строк
* @param[in]    cols        Общее количество столбцов
* @param[in]    localRows   Количество строк в блоке текущего процесса
* @param[in]    localCols   Количество столбцов в блоке текущего процесса
*/
void GatherResultsBlocks(double* localResult, double* result,
                        int commRank, int commSize,
                        int rows, int cols, int localRows, int localCols);

/**
* @brief Подсчет колонок для каждого процесса
*
* @param[in]	totalCols   Всего столбцов в матрице
* @param[in]	commSize	Общее количество процессов
* @param[in]	counts		Массив столбцов, каждому индексу массива соответсвует номер процесса
* @param[out]	displs		Массив смещений, каждому индексу массива соответсвует номер процесса
*/
void ComputeCountsDispls(int totalCols, int commSize, int** counts, int** displs);

#endif