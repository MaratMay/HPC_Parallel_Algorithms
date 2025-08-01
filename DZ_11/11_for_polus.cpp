#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#define N 10000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int remaining_rows = N % size;
    int local_rows = rank < remaining_rows ? rows_per_process + 1 : rows_per_process;

    std::vector<float> local_matrix(local_rows * N);
    std::vector<float> b(N);
    std::vector<float> local_result(local_rows);

    MPI_Win win_b, win_result;
    float* b_shared = nullptr;
    MPI_Win_allocate(rank == 0 ? N * sizeof(float) : 0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &b_shared, &win_b);

    float* result_shared = nullptr;
    MPI_Win_allocate(rank == 0 ? N * sizeof(float) : 0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &result_shared, &win_result);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    if (rank == 0) {
        for (int i = 0; i < N; i++) b[i] = distribution(generator);
        std::copy(b.begin(), b.end(), b_shared);
    }

    for (int i = 0; i < local_rows * N; i++) local_matrix[i] = distribution(generator);

    double start_time = MPI_Wtime();

    MPI_Win_fence(0, win_b);

    MPI_Get(b.data(), N, MPI_FLOAT, 0, 0, N, MPI_FLOAT, win_b);
    MPI_Win_fence(0, win_b);

    for (int i = 0; i < local_rows; i++) 
        for (int j = 0; j < N; j++) 
            local_result[i] += local_matrix[i * N + j] * b[j];

    MPI_Win_fence(0, win_result);

    int offset = 0;
    for (int i = 0; i < rank; i++) offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
    MPI_Put(local_result.data(), local_rows, MPI_FLOAT, 0, offset, local_rows, MPI_FLOAT, win_result);

    MPI_Win_fence(0, win_result);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Количество процессов: " << size << std::endl;
        std::cout << "Время выполнения: " << max_time << " секунд" << std::endl;
    }

    MPI_Win_free(&win_b);
    MPI_Win_free(&win_result);
    MPI_Finalize();
    return 0;
}