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
    ASSERT_EQ(std::filesystem::exists(f), true);
}

// TEST(OpenCLTest, Debug) {
//     Eigen::MatrixXf a(2, 2);
//     a << 1, 2, 2, 1;
//     Eigen::MatrixXf b(2, 2);
//     b << 2, 3, 1, 4;


//     int widthA  = a.cols();
//     int heightA = a.rows();
//     int widthB  = b.cols();
//     int heightB = b.rows();

//     // Create 1D views of the matrices so we can work on it
//     auto A = a.reshaped<Eigen::RowMajor>();
//     auto B = b.reshaped<Eigen::RowMajor>();

//     cl::Context context(CL_DEVICE_TYPE_DEFAULT);
//     cl::CommandQueue queue(context);
//     cl::Program program = matmul::cl_utils::build_program(context, "matmul");
//     cl::KernelFunctor<cl::Buffer, int, int, int, int, cl::Buffer, cl::Buffer>
//         matmul(program, "matmul");

//     // Create buffers (device memory)
//     cl::Buffer a_device(context, A.begin(), A.end(), /*Read only*/ true);
//     cl::Buffer b_device(context, B.begin(), B.end(), /*Read only*/ true);
//     cl::Buffer c_device(
//         context, CL_MEM_WRITE_ONLY, sizeof(float) * heightA * widthB);

//     // Perform the computation
//     matmul(cl::EnqueueArgs(queue, cl::NDRange(widthB, heightA)), c_device,
//         widthA, heightA, widthB, heightB, a_device, b_device);

//     // Copy the result from the device to the host
//     Eigen::MatrixXf c(2, 2);
//     auto C = c.reshaped<Eigen::RowMajor>();
//     cl::copy(queue, c_device, C.begin(), C.end());

//     ASSERT_EQ(c, a * b);
// }

TEST(OpenCLTest, Base) {
    Eigen::MatrixXf a(2, 2);
    a << 1, 2, 2, 1;
    Eigen::MatrixXf b(2, 2);
    b << 2, 3, 1, 4;
    Eigen::MatrixXf r(2, 2);
    r << 4, 11, 5, 10;
    ASSERT_EQ(matmul::opencl(a, b), r);
}