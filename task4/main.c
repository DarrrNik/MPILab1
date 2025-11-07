#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int NX = 100;
int NY = 100;
#define MAX_ITERS 5000
#define TOL 1e-6

#define IDX(i,j,ny) ((i)*(ny) + (j))

double f_source(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc != 3) {
        if (my_rank == 0) printf("Usage: %s <NX> <NY>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    NX = atoi(argv[1]);
    NY = atoi(argv[2]);

    int PX = (int)sqrt(comm_sz);
    int PY = PX;
    if (PX * PY != comm_sz) {
        if (my_rank == 0) fprintf(stderr, "Number of processes must be a square\n");
        MPI_Finalize();
        return 1;
    }

    int rank_x = my_rank / PY;
    int rank_y = my_rank % PY;

    if (NX % PX != 0 || NY % PY != 0) {
        if (my_rank == 0) fprintf(stderr, "NX must be divisible by PX, NY by PY\n");
        MPI_Finalize();
        return 1;
    }
    
    int nx_local = NX / PX;
    int ny_local = NY / PY;
    int nx = nx_local + 2;
    int ny = ny_local + 2;

    double h = 1.0 / (NX + 1.0);

    double *u_old = malloc(nx * ny * sizeof(double));
    double *u_new = malloc(nx * ny * sizeof(double));
    
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            u_old[IDX(i,j,ny)] = 0.0;
            u_new[IDX(i,j,ny)] = 0.0;
        }
    }
    
    double bc_val = 100.0;
    if (rank_x == 0) {
        for (int j = 0; j < ny; ++j) {
            u_old[IDX(0,j,ny)] = bc_val;
            u_new[IDX(0,j,ny)] = bc_val;
        }
    }
    if (rank_x == PX - 1) {
        for (int j = 0; j < ny; ++j) {
            u_old[IDX(nx-1,j,ny)] = bc_val;
            u_new[IDX(nx-1,j,ny)] = bc_val;
        }
    }
    if (rank_y == 0) {
        for (int i = 0; i < nx; ++i) {
            u_old[IDX(i,0,ny)] = bc_val;
            u_new[IDX(i,0,ny)] = bc_val;
        }
    }
    if (rank_y == PY - 1) {
        for (int i = 0; i < nx; ++i) {
            u_old[IDX(i,ny-1,ny)] = bc_val;
            u_new[IDX(i,ny-1,ny)] = bc_val;
        }
    }

    MPI_Datatype column_type;
    MPI_Type_vector(nx_local, 1, ny, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    int up    = (rank_x > 0)       ? my_rank - PY : MPI_PROC_NULL;
    int down  = (rank_x < PX - 1)  ? my_rank + PY : MPI_PROC_NULL;
    int left  = (rank_y > 0)       ? my_rank - 1  : MPI_PROC_NULL;
    int right = (rank_y < PY - 1)  ? my_rank + 1  : MPI_PROC_NULL;

    double global_diff = 1e100;
    int iter;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (iter = 0; iter < MAX_ITERS; ++iter) {
        for (int i = 1; i <= nx_local; ++i) {
            for (int j = 1; j <= ny_local; ++j) {
                u_old[IDX(i,j,ny)] = u_new[IDX(i,j,ny)];
            }
        }

        int W = PX + PY - 2;
        
        for (int wave = 0; wave <= W; ++wave) {
            if (rank_x + rank_y == wave) {
                if (down != MPI_PROC_NULL) {
                    MPI_Send(&u_old[IDX(nx_local,1,ny)], ny_local, MPI_DOUBLE, down, 
                             iter*100 + wave, MPI_COMM_WORLD);
                }
                if (right != MPI_PROC_NULL) {
                    MPI_Send(&u_old[IDX(1, ny_local, ny)], 1, column_type, right, 
                             iter*100 + wave, MPI_COMM_WORLD);
                }
            }
            
            if (rank_x + rank_y == wave + 1) {
                if (up != MPI_PROC_NULL) {
                    MPI_Recv(&u_new[IDX(0,1,ny)], ny_local, MPI_DOUBLE, up, 
                             iter*100 + wave, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (left != MPI_PROC_NULL) {
                    MPI_Recv(&u_new[IDX(1,0,ny)], 1, column_type, left, 
                             iter*100 + wave, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        }

        double local_diff = 0.0;
        for (int i = 1; i <= nx_local; ++i) {
            for (int j = 1; j <= ny_local; ++j) {
                int gi = rank_x * nx_local + (i - 1);
                int gj = rank_y * ny_local + (j - 1);
                double x = (gi + 1) * h;
                double y = (gj + 1) * h;
                double rhs = f_source(x, y);
                
                double s = u_new[IDX(i-1,j,ny)] +   
                           u_old[IDX(i+1,j,ny)] +     
                           u_new[IDX(i,j-1,ny)] +   
                           u_old[IDX(i,j+1,ny)];   
                
                double new_val = 0.25 * (s - h*h * rhs);
                local_diff += fabs(new_val - u_new[IDX(i,j,ny)]);
                u_new[IDX(i,j,ny)] = new_val;
            }
        }

        double global_diff_sq = 0.0;
        MPI_Allreduce(&local_diff, &global_diff_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_diff = sqrt(global_diff_sq);

        for (int i = 1; i <= nx_local; ++i) {
            for (int j = 1; j <= ny_local; ++j) {
                u_old[IDX(i,j,ny)] = u_new[IDX(i,j,ny)];
            }
        }

        if (my_rank == 0 && iter % 100 == 0) {
            printf("Iter %d, change = %.6e\n", iter, global_diff);
        }

        if (global_diff < TOL) {
            if (my_rank == 0) printf("Converged at iter %d, change = %.6e\n", iter, global_diff);
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    double globalTime = 0.0;

    MPI_Reduce(&elapsed, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        FILE *tf = fopen("times.txt", "a");
        if (tf) {
            fprintf(tf, "%d %d %d %d %.6f %.6e\n", NX, NY, comm_sz, (iter+1), globalTime, global_diff);
            fclose(tf);
        }
        printf("Finished: Grid=%dx%d, P=%d, iters=%d, time=%.6f s, change=%.6e\n", 
               NX, NY, comm_sz, iter+1, globalTime, global_diff);
    }

    free(u_old);
    free(u_new);
    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}