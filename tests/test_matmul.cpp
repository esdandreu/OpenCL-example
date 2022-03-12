#include "matmul.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>

int MAX_DEVICES;

TEST(EigenTest, Multiply) {
    // Input
    Eigen::MatrixXf a(2, 2);
    a << 1, 2, 2, 1;
    Eigen::MatrixXf b(2, 2);
    b << 2, 3, 1, 4;
    int heightA = a.rows();
    int widthB  = b.cols();
    int heightB = b.rows();
    auto A      = a.reshaped<Eigen::RowMajor>();
    auto B      = b.reshaped<Eigen::RowMajor>();

    // Output
    Eigen::MatrixXf c(heightA, widthB);
    auto C = c.reshaped<Eigen::RowMajor>();

    // C++ multiplication
    for (int col = 0; col < widthB; col++) {
        for (int row = 0; row < heightA; row++) {
            for (int i = 0; i < heightB; i++) {
                C[row * widthB + col] +=
                    A[row * heightB + i] * B[i * widthB + col];
            }
        }
    }

    // Hardcoded result
    Eigen::MatrixXf r(2, 2);
    r << 4, 11, 5, 10;
    ASSERT_EQ(a * b, r);
    // ASSERT_TRUE(c.isApprox(r));
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
            std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            cl_int max_sub_devices;
            device.getInfo(
                CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &max_sub_devices);
            std::cout << "\t\tMax Sub Devices: " << max_sub_devices
                      << std::endl;
            // Try to construct a matmul::opencl object with each device
            matmul::opencl clmatmul(device);
            // Print relevant information of that device
            std::cout
                << "\t\tMax Compute Units: "
                << clmatmul.device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                << std::endl
                << "\t\tMax Workgroup Size: "
                << clmatmul.device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                << std::endl
                << "\t\tKernel Work Group Size: "
                << clmatmul.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
                       clmatmul.device)
                << std::endl
                << "\t\tLocal Memory: "
                << clmatmul.device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                << std::endl
                << "\t\tGlobal Memory: "
                << clmatmul.device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                << std::endl
                << "\t\tGlobal Cache Memory: "
                << clmatmul.device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()
                << std::endl
                << "\t\tGlobal Cache Line Memory: "
                << clmatmul.device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()
                << std::endl
                << "\t\tOpenCL Supported Version: "
                << clmatmul.device.getInfo<CL_DEVICE_VERSION>() << std::endl
                << "\t\tAvailable: "
                << clmatmul.device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
            std::cout << "\t\tMax Work Item Sizes: ";
            for (auto& s :
                clmatmul.device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
                std::cout << s << " ";
            }
            std::cout << std::endl;
            // std::cout << "\t\tPartition properties: ";
            // for (auto& s :
            //     clmatmul.device.getInfo<CL_DEVICE_PARTITION_PROPERTIES>()) {
            //     std::cout << s << " ";
            // }
            // std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

TEST(OpenCLTest, ExampleHost) {
    // Define input
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    int heightA       = a.rows();
    int widthB        = b.cols();
    int heightB       = b.rows();
    // Create 1D views of the matrices so we can work on it
    auto A = a.reshaped<Eigen::RowMajor>();
    auto B = b.reshaped<Eigen::RowMajor>();

    // Define context and kernel
    cl::Context context(cl::Device::getDefault());
    cl::CommandQueue queue(context);
    cl::Program program = matmul::cl_utils::build_program(context, "matmul");
    cl::Kernel kernel(program, "matmul");
    cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer> matmul(
        kernel);

    // Create buffers (device memory)
    cl::Buffer A_device(context, A.begin(), A.end(), /*Read only*/ true);
    cl::Buffer B_device(context, B.begin(), B.end(), /*Read only*/ true);
    cl::Buffer output(
        context, CL_MEM_WRITE_ONLY, sizeof(float) * widthB * widthB);

    // Enqueue, submit and start execution
    matmul(cl::EnqueueArgs(queue, cl::NDRange(widthB, widthB)), output, widthB,
        heightB, A_device, B_device);

    // Copy the result from the device once the execution ended
    Eigen::MatrixXf c(heightA, widthB);
    auto C = c.reshaped<Eigen::RowMajor>();
    cl::copy(queue, output, C.begin(), C.end());

    ASSERT_TRUE(c.isApprox(a * b));
}

TEST(OpenCLTest, Base) {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    matmul::opencl clmatmul;
    ASSERT_TRUE(clmatmul(a, b).isApprox(a * b));
}

TEST(OpenCLTest, InvalidDevice) {
    cl::Device device;
    ASSERT_THROW(matmul::opencl clmatmul(device), cl::Error);
}

class DeviceTest : public testing::TestWithParam<std::tuple<int, int>> {
    protected:
    std::vector<cl::Device> devices;

    void SetUp() override {
        devices = matmul::cl_utils::get_all_devices();
    }
};

TEST_P(DeviceTest, WorkGroupSize) {
    auto [device_id, workgroup_size] = GetParam();
    if (device_id >= devices.size()) {
        GTEST_SKIP() << "No more devices found";
    }
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
    cl::Device device = devices[device_id];
    matmul::opencl clmatmul(device);
    ASSERT_TRUE(clmatmul(a, b, workgroup_size).isApprox(a * b));
}

INSTANTIATE_TEST_SUITE_P(OpenCLTest,
    DeviceTest,
    // For some reason, passing cl::Device directly as a parameter fails,
    // therefore we will work around it by setting the device index and
    // finding all devices in the test setup.
    testing::Combine(testing::Range(0, MAX_DEVICES), testing::Values(1, 2)),
    [](const testing::TestParamInfo<std::tuple<int, int>>& info) {
        int device_id = std::get<0>(info.param);
        int N         = std::get<1>(info.param);
        std::stringstream ss;
        ss << "D" << device_id << "N" << N;
        return ss.str();
    });

// TODO this should be a parametric test
TEST(OpenCLTest, createSubDevices) {
    std::vector<cl::Device> all_devices = matmul::cl_utils::get_all_devices();
    for (auto& device : all_devices) {
        // Check if device is valid to be partitioned
        cl_int max_sub_devices;
        device.getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &max_sub_devices);
        // std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        // std::cout << "Max Sub Devices: " << max_sub_devices << std::endl;
        if (max_sub_devices == 0) {
            continue;
        }
        // We only succeeded partitioning CPUs
        if (device.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_CPU) {
            continue;
        }
        // Partition the device
        std::vector<cl::Device> subdevices;
        cl_device_partition_property properties[] = {
            CL_DEVICE_PARTITION_BY_COUNTS, 3,
            CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0
        }; // 0 terminates the property list
        device.createSubDevices(properties, &subdevices);
        ASSERT_EQ(subdevices.size(), 1);
        EXPECT_EQ(subdevices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(), 3);
        for (auto& subdevice : subdevices) {
            // Check that the partition is valid
            matmul::opencl clmatmul(subdevice);
            Eigen::MatrixXf a = Eigen::MatrixXf::Random(2, 2);
            Eigen::MatrixXf b = Eigen::MatrixXf::Random(2, 2);
            ASSERT_TRUE(clmatmul(a, b).isApprox(a * b));
        }
    }
}

int main(int argc, char** argv) {
    MAX_DEVICES = matmul::cl_utils::get_all_devices().size();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}