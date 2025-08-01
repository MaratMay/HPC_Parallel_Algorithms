#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#define N 20000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims); // Определяем размер процессной решётки
    int rows = dims[0], cols = dims[1]; 

    // Координаты текущего процесса в решётке
    int coords[2];
    MPI_Comm grid_comm;
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int block_rows = N / rows;
    int block_cols = N / cols;

    std::vector<float> local_matrix(block_rows * block_cols);
    std::vector<float> local_result(block_rows, 0.0f);
    std::vector<float> b(N);

    // Создаём окно для вектора b
    MPI_Win win_b;
    float* b_shared = nullptr;
    MPI_Win_allocate(rank == 0 ? N * sizeof(float) : 0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &b_shared, &win_b);

    MPI_Win win_result;
    float* result_shared = nullptr;
    MPI_Win_allocate(rank == 0 ? N * sizeof(float) : 0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &result_shared, &win_result);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rank;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < block_rows * block_cols; i++) local_matrix[i] = distribution(generator);

    if (rank == 0) {
        for (int i = 0; i < N; i++) b[i] = distribution(generator);
        std::copy(b.begin(), b.end(), b_shared); // Копируем вектор b в окно
    }

    double start_time = MPI_Wtime();
    MPI_Win_fence(0, win_b);
    MPI_Get(b.data(), N, MPI_FLOAT, 0, 0, N, MPI_FLOAT, win_b);
    MPI_Win_fence(0, win_b);

    for (int i = 0; i < block_rows; i++)
        for (int j = 0; j < block_cols; j++)
            local_result[i] += local_matrix[i * block_cols + j] * b[coords[1] * block_cols + j];

    MPI_Win_fence(0, win_result);
    int offset = coords[0] * block_rows;
    MPI_Put(local_result.data(), block_rows, MPI_FLOAT, 0, offset, block_rows, MPI_FLOAT, win_result);
    MPI_Win_fence(0, win_result);

    // Завершаем замер времени
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