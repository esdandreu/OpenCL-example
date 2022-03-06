#include "matmul.hpp"
#include <iostream>

matmul::opencl::opencl() {
    // Initialize the function
    // TODO assign number of cores and device
    context = cl::Context(CL_DEVICE_TYPE_DEFAULT);
    queue = cl::CommandQueue(context);
    program = matmul::cl_utils::build_program(context, "matmul");
    std::cout << "Build program" << std::endl;
}

Eigen::MatrixXf
matmul::opencl::operator()(Eigen::MatrixXf& a, Eigen::MatrixXf& b) {
    int heightA = a.rows();
    int widthB  = b.cols();
    int heightB = b.rows();

    // Create 1D views of the matrices so we can work on it
    auto A = a.reshaped<Eigen::RowMajor>();
    auto B = b.reshaped<Eigen::RowMajor>();

    // Our program
    cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer> matmul(
        program, "matmul");

    // Create buffers (device memory)
    // ! Can't read from A
    // cl::Buffer A_device(context, CL_MEM_READ_ONLY, sizeof(a), a.data());
    cl::Buffer A_device(context, A.begin(), A.end(), /*Read only*/ true);

    // Seems to be equivalent and we skip the reshape
    // cl::Buffer B_device(context, CL_MEM_READ_ONLY, sizeof(b), b.data());
    cl::Buffer B_device(context, B.begin(), B.end(), /*Read only*/ true);
    cl::Buffer output(
        context, CL_MEM_WRITE_ONLY, sizeof(float) * widthB * widthB);

    // Perform the computation
    matmul(cl::EnqueueArgs(queue, cl::NDRange(widthB, widthB)), output, widthB,
        heightB, A_device, B_device);

    // Copy the result from the device to the host
    Eigen::MatrixXf c(heightA, widthB);
    auto C = c.reshaped<Eigen::RowMajor>();
    cl::copy(queue, output, C.begin(), C.end());
    return c;
}
