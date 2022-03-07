#include "matmul.hpp"
#include <benchmark/benchmark.h>
#include <iostream>

int MAX_DEVICES;

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
    std::vector<cl::Device> devices;
    cl::Device device;
    int workgroup_size;

    ClMatMul() {
        devices = matmul::cl_utils::get_all_devices();
        int i   = 0;
        for (auto& device : devices) {
            std::cout << i++ << ": " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
        }
    }

    void SetUp(const ::benchmark::State& state) {
        MatMul::SetUp(state);
        device         = devices[state.range(1)];
        clmatmul       = matmul::opencl(device);
        // TODO check if workgroup_size is too big
        workgroup_size = state.range(2);
    }
};

BENCHMARK_DEFINE_F(ClMatMul, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b, workgroup_size);
    }
}

BENCHMARK_REGISTER_F(ClMatMul, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({ benchmark::CreateRange(8, 1024, /*mult*/ 2),
        benchmark::CreateDenseRange(0, 2, /*step*/ 1),
        benchmark::CreateRange(1, 8, /*mult*/ 2) });

int main(int argc, char** argv) {
    MAX_DEVICES = matmul::cl_utils::get_all_devices().size();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}