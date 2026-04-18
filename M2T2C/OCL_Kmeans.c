

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <CL/cl.h>

#define MAX_ITER   50
#define NUM_DIM    2
#define SEED       42
#define LOCAL_SIZE 64

static const char *kernel_src =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void kmeans_assign(\n"
"    __global const double *points,\n"
"    __global const double *centroids,\n"
"    __global int *labels,\n"
"    const int my_n,\n"
"    const int K)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i >= my_n) return;\n"
"\n"
"    double px = points[i * 2 + 0];\n"
"    double py = points[i * 2 + 1];\n"
"    double best_dist = 1.0e300;\n"
"    int best_k = 0;\n"
"\n"
"    for (int k = 0; k < K; k++) {\n"
"        double dx = px - centroids[k * 2 + 0];\n"
"        double dy = py - centroids[k * 2 + 1];\n"
"        double d = dx * dx + dy * dy;\n"
"        if (d < best_dist) {\n"
"            best_dist = d;\n"
"            best_k = k;\n"
"        }\n"
"    }\n"
"\n"
"    labels[i] = best_k;\n"
"}\n";

static double dist_sq(const double *p, const double *c) {
    double dx = p[0] - c[0];
    double dy = p[1] - c[1];
    return dx * dx + dy * dy;
}

static const char *device_type_name(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU) return "GPU";
    if (t & CL_DEVICE_TYPE_CPU) return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "OTHER";
}

static int select_opencl_device(cl_platform_id *platform_out,
                                cl_device_id *device_out,
                                cl_device_type *type_out) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return -1;
    }

    cl_platform_id *platforms =
        (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        return -1;
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return -1;
    }

    const cl_device_type try_types[3] = {
        CL_DEVICE_TYPE_GPU,
        CL_DEVICE_TYPE_CPU,
        CL_DEVICE_TYPE_ALL
    };

    for (cl_uint p = 0; p < num_platforms; p++) {
        for (int t = 0; t < 3; t++) {
            cl_device_id dev = NULL;
            err = clGetDeviceIDs(platforms[p], try_types[t], 1, &dev, NULL);
            if (err == CL_SUCCESS && dev != NULL) {
                *platform_out = platforms[p];
                *device_out = dev;
                *type_out = try_types[t];
                free(platforms);
                return 0;
            }
        }
    }

    free(platforms);
    return -1;
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

    double *all_points_d = NULL;
    double *centroids_d   = (double *)malloc(K * NUM_DIM * sizeof(double));
    double *local_points_d = (double *)malloc(alloc_n * NUM_DIM * sizeof(double));

    double *local_sum   = (double *)calloc(K * NUM_DIM, sizeof(double));
    int    *local_count  = (int *)calloc(K, sizeof(int));
    double *global_sum   = (double *)calloc(K * NUM_DIM, sizeof(double));
    int    *global_count = (int *)calloc(K, sizeof(int));

    int *labels = (int *)malloc(alloc_n * sizeof(int));

    if (rank == 0) {
        all_points_d = (double *)malloc(N * NUM_DIM * sizeof(double));
        srand(SEED);

        for (int i = 0; i < N * NUM_DIM; i++) {
            all_points_d[i] = (double)rand() / (double)RAND_MAX * 100.0;
        }

        for (int k = 0; k < K; k++) {
            centroids_d[k * NUM_DIM + 0] = all_points_d[k * NUM_DIM + 0];
            centroids_d[k * NUM_DIM + 1] = all_points_d[k * NUM_DIM + 1];
        }
    }

    MPI_Bcast(centroids_d, K * NUM_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(all_points_d, sendcounts, displs, MPI_DOUBLE,
                 local_points_d, my_n * NUM_DIM, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_device_type requested_type = CL_DEVICE_TYPE_ALL;

    if (select_opencl_device(&platform, &device, &requested_type) != 0) {
        if (rank == 0) {
            printf("No OpenCL platform/device found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    char platform_name[128] = {0};
    char device_name[128] = {0};
    cl_device_type actual_type = 0;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                      sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME,
                    sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE,
                    sizeof(actual_type), &actual_type, NULL);

    printf("Rank %d OpenCL platform: %s\n", rank, platform_name);
    printf("Rank %d OpenCL device: %s [%s]\n",
           rank, device_name, device_type_name(actual_type));
    printf("Rank %d kernel config: global_size rounded to multiple of local_size, local_size=%d, local_n=%d\n",
           rank, LOCAL_SIZE, my_n);

    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };

    cl_context ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || ctx == NULL) {
        fprintf(stderr, "Rank %d: clCreateContext failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS || queue == NULL) {
        fprintf(stderr, "Rank %d: clCreateCommandQueue failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    if (err != CL_SUCCESS || prog == NULL) {
        fprintf(stderr, "Rank %d: clCreateProgramWithSource failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Rank %d OpenCL build log:\n%s\n", rank, log);
            free(log);
        }
        fprintf(stderr, "Rank %d: clBuildProgram failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_kernel kernel = clCreateKernel(prog, "kmeans_assign", &err);
    if (err != CL_SUCCESS || kernel == NULL) {
        fprintf(stderr, "Rank %d: clCreateKernel failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t point_bytes = (size_t)alloc_n * NUM_DIM * sizeof(double);
    size_t centroid_bytes = (size_t)K * NUM_DIM * sizeof(double);

    cl_mem d_points = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                     point_bytes, NULL, &err);
    if (err != CL_SUCCESS || d_points == NULL) {
        fprintf(stderr, "Rank %d: clCreateBuffer d_points failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                        centroid_bytes, NULL, &err);
    if (err != CL_SUCCESS || d_centroids == NULL) {
        fprintf(stderr, "Rank %d: clCreateBuffer d_centroids failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_mem d_labels = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                     (size_t)alloc_n * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS || d_labels == NULL) {
        fprintf(stderr, "Rank %d: clCreateBuffer d_labels failed (%d)\n", rank, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (my_n > 0) {
        err = clEnqueueWriteBuffer(queue, d_points, CL_TRUE, 0,
                                   (size_t)my_n * NUM_DIM * sizeof(double),
                                   local_points_d, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Rank %d: write d_points failed (%d)\n", rank, err);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_points);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_labels);
    clSetKernelArg(kernel, 3, sizeof(int), &my_n);
    clSetKernelArg(kernel, 4, sizeof(int), &K);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        MPI_Bcast(centroids_d, K * NUM_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        err = clEnqueueWriteBuffer(queue, d_centroids, CL_TRUE, 0,
                                   centroid_bytes, centroids_d,
                                   0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Rank %d: write d_centroids failed (%d)\n", rank, err);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        memset(local_sum, 0, K * NUM_DIM * sizeof(double));
        memset(local_count, 0, K * sizeof(int));

        size_t local_size = (my_n > 0 && my_n < LOCAL_SIZE) ? (size_t)my_n : (size_t)LOCAL_SIZE;
        if (local_size == 0) local_size = 1;

        size_t global_size = ((size_t)my_n + local_size - 1) / local_size * local_size;
        if (global_size == 0) global_size = local_size;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                     &global_size, &local_size,
                                     0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Rank %d: clEnqueueNDRangeKernel failed (%d)\n", rank, err);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        clFinish(queue);

        if (my_n > 0) {
            err = clEnqueueReadBuffer(queue, d_labels, CL_TRUE, 0,
                                      (size_t)my_n * sizeof(int),
                                      labels, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Rank %d: read d_labels failed (%d)\n", rank, err);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        for (int i = 0; i < my_n; i++) {
            int k = labels[i];
            local_sum[k * NUM_DIM + 0] += local_points_d[i * NUM_DIM + 0];
            local_sum[k * NUM_DIM + 1] += local_points_d[i * NUM_DIM + 1];
            local_count[k]++;
        }

        MPI_Allreduce(local_sum, global_sum, K * NUM_DIM,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(local_count, global_count, K,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int k = 0; k < K; k++) {
                if (global_count[k] > 0) {
                    centroids_d[k * NUM_DIM + 0] =
                        global_sum[k * NUM_DIM + 0] / global_count[k];
                    centroids_d[k * NUM_DIM + 1] =
                        global_sum[k * NUM_DIM + 1] / global_count[k];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== MPI + OpenCL K-Means Clustering ===\n");
        printf("Points=%d | K=%d | Iterations=%d | Processes=%d\n",
               N, K, MAX_ITER, size);
        printf("Time=%.4f sec\n", t_end - t_start);
        printf("Final centroids (first 5 shown):\n");

        int show = (K < 5) ? K : 5;
        for (int k = 0; k < show; k++) {
            printf("  C[%d] = (%.4f, %.4f)  count=%d\n",
                   k,
                   centroids_d[k * NUM_DIM + 0],
                   centroids_d[k * NUM_DIM + 1],
                   global_count[k]);
        }

        printf("Correctness check: compare these centroids with mpi_kmeans output.\n");
    }

    clReleaseMemObject(d_points);
    clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_labels);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(sendcounts);
    free(displs);
    free(centroids_d);
    free(local_points_d);
    free(local_sum);
    free(local_count);
    free(global_sum);
    free(global_count);
    free(labels);
    if (rank == 0) free(all_points_d);

    MPI_Finalize();
    return 0;
}
