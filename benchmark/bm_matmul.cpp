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

class ClMatMul : public MatMul {
    public:
    cl_device_type type;
    matmul::opencl clmatmul;
    std::vector<cl::Device> devices;
    cl::Device device;

    ClMatMul(cl_device_type type = CL_DEVICE_TYPE_ALL) : type(type) {
        devices = matmul::cl_utils::get_all_devices(type);
        int i   = 0;
        for (auto& device : devices) {
            std::cout << i++ << ": " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
        }
    }

    cl::Device getDevice(benchmark::State& state) {
        if (state.range(1) >= devices.size()) {
            state.SkipWithError("Invalid device index");
        }
        cl::Device d = devices[state.range(1)];
        if (d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() < 2) {
            state.SkipWithError("Device does not support two dimmensions");
        }
        return d;
    }

    void SetUp(benchmark::State& state) {
        MatMul::SetUp(state);
        device   = getDevice(state);
        clmatmul = matmul::opencl(device);
    }
};

BENCHMARK_DEFINE_F(ClMatMul, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b);
    }
}

BENCHMARK_REGISTER_F(ClMatMul, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({ /*Work*/ benchmark::CreateRange(256, 8 << 10, /*mult*/ 2),
        /*Devices*/ benchmark::CreateDenseRange(0, 2, /*step*/ 1) });

class ClMatMulComputeUnits : public ClMatMul {

    public:
    int compute_units;

    ClMatMulComputeUnits(cl_device_type type = CL_DEVICE_TYPE_CPU)
    : ClMatMul(type){};

    void SetUp(benchmark::State& state) {
        MatMul::SetUp(state);
        device = getDevice(state);
        std::vector<cl::Device> subdevices;
        compute_units = state.range(2);
        if (compute_units > device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()) {
            state.SkipWithError("Device has not so many compute units");
        }
        cl_device_partition_property properties[] = {
            CL_DEVICE_PARTITION_BY_COUNTS, compute_units,
            CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0
        }; // 0 terminates the property list
        device.createSubDevices(properties, &subdevices);
        if (subdevices.size() < 1) {
            state.SkipWithError("Could not create subdevices");
        }
        clmatmul = matmul::opencl(subdevices[0]);
    }
};

BENCHMARK_DEFINE_F(ClMatMulComputeUnits, OpenCL)(benchmark::State& state) {
    for (auto _ : state) {
        clmatmul(a, b);
    }
}

BENCHMARK_REGISTER_F(ClMatMulComputeUnits, OpenCL)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({ /*Work*/ benchmark::CreateRange(1 << 9, 1 << 21, /*mult*/ 8),
        /*Devices*/ { 0 },
        /*ComputeUnits*/ benchmark::CreateDenseRange(1, 12, /*step*/ 1) });

int main(int argc, char** argv) {
    MAX_DEVICES = matmul::cl_utils::get_all_devices().size();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}