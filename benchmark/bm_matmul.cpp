#include "matmul.hpp"
#include <benchmark/benchmark.h>
#include <iostream>

class MatMul : public benchmark::Fixture {
    public:
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;

    void SetUp(const ::benchmark::State& state) {
        int size = state.range(0);
        a        = Eigen::MatrixXf::Random(size, size);
        b        = Eigen::MatrixXf::Random(size, size);
    }
};

BENCHMARK_DEFINE_F(MatMul, Eigen)(benchmark::State& state) {
    for (auto _ : state) {
        auto c = a * b;
        c.eval();
    }
}
BENCHMARK_REGISTER_F(MatMul, Eigen)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(8, 512);

class ClMatMul : public MatMul {
    public:
    matmul::opencl clmatmul;

    void SetUp(const ::benchmark::State& state) {
        MatMul::SetUp(state);
        clmatmul = matmul::opencl();
    }
};

BENCHMARK_DEFINE_F(ClMatMul, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b);
    }
}
BENCHMARK_REGISTER_F(ClMatMul, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(8, 1024);
