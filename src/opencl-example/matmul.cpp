#include "matmul.hpp"

int add(int a, int b) {
    return a + b;
}

Eigen::MatrixXf
matmul::opencl(const Eigen::MatrixXf a, const Eigen::MatrixXf b) {
    int heightA = a.rows();
    int widthB  = b.cols();
    int heightB = b.rows();

    // Create 1D views of the matrices so we can work on it
    auto A = a.reshaped<Eigen::RowMajor>();
    auto B = b.reshaped<Eigen::RowMajor>();

    // Create context
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    cl::CommandQueue queue(context);
    cl::Program program = matmul::cl_utils::build_program(context, "matmul");
    cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer> matmul(
        program, "matmul");

    // Create buffers (device memory)
    cl::Buffer A_device(context, A.begin(), A.end(), /*Read only*/ true);
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
