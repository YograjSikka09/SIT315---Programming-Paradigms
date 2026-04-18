
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#define MAX_ITER 50
#define NUM_DIM  2
#define SEED     42

static double dist_sq(const double *p, const double *c) {
    double dx = p[0] - c[0];
    double dy = p[1] - c[1];
    return dx * dx + dy * dy;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <num_points> <num_clusters>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if (N <= 0 || K <= 0) {
        if (rank == 0) {
            printf("N and K must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int base  = N / size;
    int extra = N % size;
    int my_n  = base + (rank < extra ? 1 : 0);

    int alloc_n = (my_n > 0) ? my_n : 1;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
        int cnt = (base + (i < extra ? 1 : 0)) * NUM_DIM;
        sendcounts[i] = cnt;
        displs[i] = offset;
        offset += cnt;
    }

    double *all_points  = NULL;
    double *centroids   = (double *)malloc(K * NUM_DIM * sizeof(double));
    double *local_points = (double *)malloc(alloc_n * NUM_DIM * sizeof(double));

    double *local_sum   = (double *)calloc(K * NUM_DIM, sizeof(double));
    int    *local_count  = (int *)calloc(K, sizeof(int));
    double *global_sum   = (double *)calloc(K * NUM_DIM, sizeof(double));
    int    *global_count = (int *)calloc(K, sizeof(int));
    int    *labels       = (int *)malloc(alloc_n * sizeof(int));

    if (rank == 0) {
        all_points = (double *)malloc(N * NUM_DIM * sizeof(double));
        srand(SEED);

        for (int i = 0; i < N * NUM_DIM; i++) {
            all_points[i] = (double)rand() / (double)RAND_MAX * 100.0;
        }

        for (int k = 0; k < K; k++) {
            centroids[k * NUM_DIM + 0] = all_points[k * NUM_DIM + 0];
            centroids[k * NUM_DIM + 1] = all_points[k * NUM_DIM + 1];
        }
    }

    MPI_Bcast(centroids, K * NUM_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(all_points, sendcounts, displs, MPI_DOUBLE,
                 local_points, my_n * NUM_DIM, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        MPI_Bcast(centroids, K * NUM_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        memset(local_sum, 0, K * NUM_DIM * sizeof(double));
        memset(local_count, 0, K * sizeof(int));

        for (int i = 0; i < my_n; i++) {
            double best_dist = DBL_MAX;
            int best_k = 0;

            for (int k = 0; k < K; k++) {
                double d = dist_sq(&local_points[i * NUM_DIM],
                                   &centroids[k * NUM_DIM]);
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }

            labels[i] = best_k;
            local_sum[best_k * NUM_DIM + 0] += local_points[i * NUM_DIM + 0];
            local_sum[best_k * NUM_DIM + 1] += local_points[i * NUM_DIM + 1];
            local_count[best_k]++;
        }

        MPI_Allreduce(local_sum, global_sum, K * NUM_DIM,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(local_count, global_count, K,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int k = 0; k < K; k++) {
                if (global_count[k] > 0) {
                    centroids[k * NUM_DIM + 0] =
                        global_sum[k * NUM_DIM + 0] / global_count[k];
                    centroids[k * NUM_DIM + 1] =
                        global_sum[k * NUM_DIM + 1] / global_count[k];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("=== MPI K-Means Clustering ===\n");
        printf("Points=%d | K=%d | Iterations=%d | Processes=%d\n",
               N, K, MAX_ITER, size);
        printf("Time=%.4f sec\n", t_end - t_start);
        printf("Final centroids (first 5 shown):\n");

        int show = (K < 5) ? K : 5;
        for (int k = 0; k < show; k++) {
            printf("  C[%d] = (%.4f, %.4f)  count=%d\n",
                   k,
                   centroids[k * NUM_DIM + 0],
                   centroids[k * NUM_DIM + 1],
                   global_count[k]);
        }
    }

    free(sendcounts);
    free(displs);
    free(centroids);
    free(local_points);
    free(local_sum);
    free(local_count);
    free(global_sum);
    free(global_count);
    free(labels);
    if (rank == 0) free(all_points);

    MPI_Finalize();
    return 0;
}
