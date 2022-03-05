#include <benchmark/benchmark.h>
#include "matmul.hpp"

static void BM_OpenCL(benchmark::State& state) {
    Eigen::MatrixXf a(2, 2);
    a << 1, 2, 3, 1;
    Eigen::MatrixXf b(2, 2);
    b << 2, 4, 1, 4;
    for (auto _ : state)
        matmul::opencl(a, b);
}
BENCHMARK(BM_OpenCL);

static void BM_Eigen(benchmark::State& state) {
    Eigen::MatrixXf a(2, 2);
    a << 1, 2, 3, 1;
    Eigen::MatrixXf b(2, 2);
    b << 2, 4, 1, 4;
    for (auto _ : state)
        matmul::opencl(a, b);
}
BENCHMARK(BM_Eigen);