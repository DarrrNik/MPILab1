#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

double* allocate_matrix(int size) {
    return (double*)malloc(size * sizeof(double));
}

void free_matrix(double* matrix) {
    if (matrix != NULL) free(matrix);
}

void initialize_matrix(double* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100.0;
    }
}

double sequential_matrix_multiply(double* A, double* B, double* C, int n) {
    double start_time = MPI_Wtime();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                volatile double a_val = A[i * n + k];
                volatile double b_val = B[k * n + j];
                sum += a_val * b_val;
            }
            C[i * n + j] = sum;
        }
    }
    
    double end_time = MPI_Wtime();
    return end_time - start_time;
}

double cannon_matrix_multiply(int n, int grid_size, int my_rank) {
    int block_size = n / grid_size;
    int block_elements = block_size * block_size;
    
    double* local_A = allocate_matrix(block_elements);
    double* local_B = allocate_matrix(block_elements);
    double* local_C = allocate_matrix(block_elements);
    
    for (int i = 0; i < block_elements; i++) {
        local_A[i] = 0.0;
        local_B[i] = 0.0;
        local_C[i] = 0.0;
    }
    
    int row = my_rank / grid_size;
    int col = my_rank % grid_size;
    
    unsigned int seed = time(NULL) + my_rank * 1000;
    srand(seed);
    
    initialize_matrix(local_A, block_elements);
    initialize_matrix(local_B, block_elements);
    
    int left = (col - 1 + grid_size) % grid_size;
    int right = (col + 1) % grid_size;
    int up = (row - 1 + grid_size) % grid_size;
    int down = (row + 1) % grid_size;
    
    int left_rank = row * grid_size + left;
    int right_rank = row * grid_size + right;
    int up_rank = up * grid_size + col;
    int down_rank = down * grid_size + col;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    for (int shift = 0; shift < row; shift++) {
        MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE,
                           left_rank, 0, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    for (int shift = 0; shift < col; shift++) {
        MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE,
                           up_rank, 1, down_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    for (int step = 0; step < grid_size; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < block_size; k++) {
                    double a_val = local_A[i * block_size + k];
                    double b_val = local_B[k * block_size + j];
                    sum += a_val * b_val;
                }
                local_C[i * block_size + j] += sum;
            }
        }
        
        if (step < grid_size - 1) {
            MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE,
                               left_rank, 2, right_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE,
                               up_rank, 3, down_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;
    double globalTime = 0.0;

    MPI_Reduce(&parallel_time, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        double* result = allocate_matrix(n * n);
        
        for (int i = 0; i < n * n; i++) {
            result[i] = 0.0;
        }
        
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                result[(row * block_size + i) * n + (col * block_size + j)] = 
                    local_C[i * block_size + j];
            }
        }
        
        for (int proc = 1; proc < grid_size * grid_size; proc++) {
            double* temp_block = allocate_matrix(block_elements);
            MPI_Recv(temp_block, block_elements, MPI_DOUBLE, proc, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int src_row = proc / grid_size;
            int src_col = proc % grid_size;
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    result[(src_row * block_size + i) * n + (src_col * block_size + j)] = 
                        temp_block[i * block_size + j];
                }
            }
            
            free_matrix(temp_block);
        }
        
        free_matrix(result);
        
    } else {
        MPI_Send(local_C, block_elements, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
    
    free_matrix(local_A);
    free_matrix(local_B);
    free_matrix(local_C);
    
    return parallel_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if (argc != 2) {
        if (my_rank == 0) {
            printf("Usage: mpiexec -np <processes> ./main <matrix_size>\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int n = atoi(argv[1]);
    int grid_size = (int)sqrt(comm_sz);
    
    if (grid_size * grid_size != comm_sz) {
        if (my_rank == 0) {
            fprintf(stderr, "Error: num procs must be fully square\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (n % grid_size != 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Error: matrix size %d doesn't divide into grid size %d\n", n, grid_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    double parallel_time = 0.0;
    double sequential_time = 0.0;
    
    if (my_rank == 0) {
        printf("=== Matrix %dx%d, Procs: %d ===\n", n, n, comm_sz);
    }
    
    parallel_time = cannon_matrix_multiply(n, grid_size, my_rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        printf("Parallel time: %.6f s\n", parallel_time);
        
        double* A = allocate_matrix(n * n);
        double* B = allocate_matrix(n * n);
        double* C = allocate_matrix(n * n);
        
        srand(42);
        initialize_matrix(A, n * n);
        initialize_matrix(B, n * n);
        
        printf("Running sequential multiply...\n");
        sequential_time = sequential_matrix_multiply(A, B, C, n);
        printf("Sequential time: %.6f s\n", sequential_time);
        
        double acceleration = sequential_time / parallel_time;
        double efficiency = (acceleration / comm_sz) * 100;
        
        printf("Acceleration: %.4f\n", acceleration);
        printf("Efficienty: %.4f%%\n", efficiency);
        
        printf("CSV:%d,%d,%.6f,%.6f,%.4f,%.4f\n", 
               n, comm_sz, parallel_time, sequential_time, acceleration, efficiency);
        
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
    }
    
    MPI_Finalize();
    return 0;
}