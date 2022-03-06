#include "matmul.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

TEST(EigenTest, Multiply) {
    Eigen::MatrixXf a(2, 2);
    a << 1, 2, 2, 1;
    Eigen::MatrixXf b(2, 2);
    b << 2, 3, 1, 4;
    Eigen::MatrixXf r(2, 2);
    r << 4, 11, 5, 10;
    ASSERT_EQ(a * b, r);
}

TEST(OpenCLTest, MatmulFileExists) {
    std::string name        = "matmul";
    std::filesystem::path f = matmul::cl_utils::get_program_path(name);
    ASSERT_TRUE(std::filesystem::exists(f));
}

TEST(OpenCLTest, PrintDevices) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& p : platforms) {
        std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& device : devices) {
            std::cout
                << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl
                << "\t\tMax Compute Units: "
                << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
                << "\t\tGlobal Memory: "
                << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl
                << "\t\tMax Clock Frequency: "
                << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl
                << "\t\tMax Memory Allocation: "
                << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl
                << "\t\tLocal Memory: "
                << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl
                << "\t\tAvailable: " << device.getInfo<CL_DEVICE_AVAILABLE>()
                << std::endl;
        }
        std::cout << std::endl;
    }
}

TEST(OpenCLTest, Base) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    matmul::opencl clmatmul;
    ASSERT_TRUE(clmatmul(a, b).isApprox(a * b));
}
