#include <benchmark/benchmark.h>
#include "matmul.hpp"
#include <iostream>

class ClMatMul : public benchmark::Fixture {
    public:
    matmul::opencl clmatmul;

    // void SetUp(const ::benchmark::State& state) {
    //     clmatmul = matmul::opencl();
    //     std::cout << "Setup" << std::endl;
    // }

    ClMatMul() {
        clmatmul = matmul::opencl();
        std::cout << "Initializing OpenCL" << std::endl;
    }
};

// static void BM_OpenCL(benchmark::State& state) {
BENCHMARK_DEFINE_F(ClMatMul, BM_OpenCL)(benchmark::State& state) {
    // state.PauseTiming();
    // matmul::opencl clmatmul;
    // state.ResumeTiming();

    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    for (auto _ : state)
        clmatmul(a, b);
}
BENCHMARK_REGISTER_F(ClMatMul, BM_OpenCL)->Range(8, 8<<10);

static void BM_Eigen(benchmark::State& state) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    for (auto _ : state)
        benchmark::DoNotOptimize(a * b);
}
BENCHMARK(BM_Eigen);