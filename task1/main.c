#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int my_rank, comm_sz;
    long long total_points, local_points, total_inside = 0, local_inside = 0;
    double start_time, end_time, pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc != 2) {
        if (my_rank == 0) {
            printf("Usage: mpiexec -np <num_procs> %s <total_points>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    total_points = atoll(argv[1]);

    if (total_points < 1)
    {
        if (my_rank == 0) {
            printf("Num of points must be more than 0!");
        }
        MPI_Finalize();
        return 1;
    }

    local_points = total_points / comm_sz;
    if (my_rank == comm_sz - 1) {
        local_points = total_points - (local_points * (comm_sz - 1));
    }

    srand(time(NULL) + my_rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    for (int i = 0; i < local_points; i++)
    {
        double x = (double)rand() / (double)RAND_MAX * 2.0;
        double y = (double)rand() / (double)RAND_MAX * 2.0;

        if ((x - 1) * (x - 1) + (y - 1) * (y - 1) <= 1)
        {
            local_inside++;
        }
    }

    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();

    double localTime = end_time - start_time, globalTime = 0.0;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        pi = 4.0 * (double)total_inside / (double)total_points;

        printf("============================================\n");
        printf("Monte-Carlo method for computing Pi number\n");
        printf("============================================\n");
        printf("Comm size: %d\n", comm_sz);
        printf("Total points: %lld\n", total_points);
        printf("Points inside circle: %lld\n", total_inside);
        printf("Calculated Pi number: %.10f\n", pi);
        printf("Real Pi number: %.10f\n", M_PI);
        printf("Error: %.10f\n", fabs(pi - M_PI));
        printf("Time: %.6f seconds\n", globalTime);
        printf("============================================\n");
        
        FILE *fp = fopen("performance_results.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %lld %.6f %.10f\n", 
                    comm_sz, total_points, globalTime, pi);
            fclose(fp);
        }
    }
    
    MPI_Finalize();
    return 0;
}