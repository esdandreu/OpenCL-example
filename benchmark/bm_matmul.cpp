#include "matmul.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <iostream>

int MAX_DEVICES;

class MatMul : public benchmark::Fixture {
    public:
    int g_workgroup_size;
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;

    void SetUp(const ::benchmark::State& state) {
        g_workgroup_size = std::sqrt(state.range(0));
        a = Eigen::MatrixXf::Random(g_workgroup_size, g_workgroup_size);
        b = Eigen::MatrixXf::Random(g_workgroup_size, g_workgroup_size);
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
    ->Range(256, 8 << 10);

class ClMatMul : public MatMul {
    public:
    matmul::opencl clmatmul;
    std::vector<cl::Device> devices;
    cl::Device device;
    int l_workgroup_size;

    ClMatMul() {
        devices = matmul::cl_utils::get_all_devices();
        int i   = 0;
        for (auto& device : devices) {
            std::cout << i++ << ": " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
        }
    }

    void SetUp(benchmark::State& state) {
        MatMul::SetUp(state);
        device           = devices[state.range(1)];
        clmatmul         = matmul::opencl(device);
        l_workgroup_size = state.range(2);
        if (int mod = g_workgroup_size % l_workgroup_size) {
            state.SkipWithError("Workgroup size is not a multiple of "
                                "local workgroup size");
        }
    }
};

BENCHMARK_DEFINE_F(ClMatMul, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b, l_workgroup_size);
    }
}

BENCHMARK_REGISTER_F(ClMatMul, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({ /*Work*/ benchmark::CreateRange(256, 1 << 14, /*mult*/ 4),
        /*Devices*/ benchmark::CreateDenseRange(0, 2, /*step*/ 1),
        /*Work-Items*/ benchmark::CreateRange(1, 16, /*mult*/ 2) });

int main(int argc, char** argv) {
    MAX_DEVICES = matmul::cl_utils::get_all_devices().size();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}