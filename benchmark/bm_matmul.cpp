#include "matmul.hpp"
#include <algorithm>
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

class ClMatMul : public benchmark::Fixture {
    public:
    matmul::opencl clmatmul;
    std::vector<cl::Device> devices;
    cl::Device device;
    unsigned long l_workgroup_size;
    unsigned long g_workgroup_size;
    Eigen::MatrixXf a;
    Eigen::MatrixXf b;

    ClMatMul() {
        devices = matmul::cl_utils::get_all_devices();
        int i   = 0;
        for (auto& device : devices) {
            std::cout << i++ << ": " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
        }
    }

    void SetUp(benchmark::State& state) {
        device = devices[state.range(0)];
        if (device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() < 2) {
            state.SkipWithError("Device does not support two dimmensions");
        }
        clmatmul = matmul::opencl(device);
        size_t k_workgroup_size =
            clmatmul.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        l_workgroup_size =
            std::sqrt(static_cast<unsigned long>(k_workgroup_size));
        g_workgroup_size = l_workgroup_size * state.range(1);
        if (int mod = g_workgroup_size % l_workgroup_size) {
            std::cout << state.range(1) << std::endl;
            std::cout << l_workgroup_size << std::endl;
            std::cout << g_workgroup_size << std::endl;
            std::cout << mod << std::endl;
            state.SkipWithError("Workgroup size is not a multiple of "
                                "local workgroup size");
        }
        a = Eigen::MatrixXf::Random(g_workgroup_size, g_workgroup_size);
        b = Eigen::MatrixXf::Random(g_workgroup_size, g_workgroup_size);
    }
};

BENCHMARK_DEFINE_F(ClMatMul, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b, l_workgroup_size);
    }
}

BENCHMARK_REGISTER_F(ClMatMul, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({ /*Devices*/ benchmark::CreateDenseRange(0, 2, /*step*/ 1),
        /*Computing Units*/ benchmark::CreateRange(1, 16, /*mult*/ 2) });

int main(int argc, char** argv) {
    MAX_DEVICES = matmul::cl_utils::get_all_devices().size();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}