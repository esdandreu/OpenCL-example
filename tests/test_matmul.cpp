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

TEST(OpenCLTest, Debug) {
    // std::vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);
    // for (auto& p : platforms) {
    //     std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
    //     std::vector<cl::Device> devices;
    //     p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    //     for (auto& d : devices) {
    //         std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << std::endl;
    //     }
    // }
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    int platform_id = 0;
    int device_id   = 0;

    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    for (std::vector<cl::Platform>::iterator it = platforms.begin();
         it != platforms.end(); ++it) {
        cl::Platform platform(*it);

        std::cout << "Platform ID: " << platform_id++ << std::endl;
        std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>()
                  << std::endl;
        std::cout
            << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>()
            << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        for (std::vector<cl::Device>::iterator it2 = devices.begin();
             it2 != devices.end(); ++it2) {
            cl::Device device(*it2);

            std::cout << std::endl
                      << "\tDevice " << device_id++ << ": " << std::endl;
            std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>()
                      << std::endl;
            std::cout
                << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>()
                << std::endl;
            std::cout
                << "\t\tDevice Version: " << device.getInfo<CL_DEVICE_VERSION>()
                << std::endl;
            switch (device.getInfo<CL_DEVICE_TYPE>()) {
            case 4: std::cout << "\t\tDevice Type: GPU" << std::endl; break;
            case 2: std::cout << "\t\tDevice Type: CPU" << std::endl; break;
            default: std::cout << "\t\tDevice Type: unknown" << std::endl;
            }
            std::cout << "\t\tDevice Max Compute Units: "
                      << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                      << std::endl;
            std::cout << "\t\tDevice Global Memory: "
                      << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                      << std::endl;
            std::cout << "\t\tDevice Max Clock Frequency: "
                      << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
                      << std::endl;
            std::cout << "\t\tDevice Max Memory Allocation: "
                      << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
                      << std::endl;
            std::cout << "\t\tDevice Local Memory: "
                      << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                      << std::endl;
            std::cout << "\t\tDevice Available: "
                      << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
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
