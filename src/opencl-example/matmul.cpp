#include "matmul.hpp"
#include <iostream>

matmul::opencl::opencl(cl::Device& device) : device(device) {
    try {
        context = cl::Context(device);
        queue   = cl::CommandQueue(context);
        program = matmul::cl_utils::build_program(context, "matmul");
        kernel  = cl::Kernel(program, "matmul");
    } catch (cl::Error error) {
        // https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
        std::cerr << "OpenCL error: " << error.what() << "(" << error.err()
                  << ")" << std::endl;
        throw error;
    }
}

Eigen::MatrixXf matmul::opencl::operator()(Eigen::MatrixXf& a,
    Eigen::MatrixXf& b,
    int workgroup_size) {
    int heightA = a.rows();
    int widthB  = b.cols();
    int heightB = b.rows();

    // Create 1D views of the matrices so we can work on it
    auto A = a.reshaped<Eigen::RowMajor>();
    auto B = b.reshaped<Eigen::RowMajor>();

    // Our program
    cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer> matmul(
        kernel);

    // Create buffers (device memory)
    cl::Buffer A_device(context, A.begin(), A.end(), /*Read only*/ true);
    cl::Buffer B_device(context, B.begin(), B.end(), /*Read only*/ true);
    cl::Buffer output(
        context, CL_MEM_WRITE_ONLY, sizeof(float) * widthB * widthB);

    if (workgroup_size == NULL) { // Use the default local workgroup size
        cl::EnqueueArgs enqueueArgs(queue, cl::NDRange(widthB, widthB));
        matmul(enqueueArgs, output, widthB, heightB, A_device, B_device);
    } else { // Specify the local workgroup size
        cl::EnqueueArgs enqueueArgs(/*command queue*/ queue,
            /*global workgroup size*/ cl::NDRange(widthB, widthB),
            /*local workgroup size*/ cl::NDRange(workgroup_size, workgroup_size));
        matmul(enqueueArgs, output, widthB, heightB, A_device, B_device);
    }

    // Copy the result from the device to the host
    Eigen::MatrixXf c(heightA, widthB);
    auto C = c.reshaped<Eigen::RowMajor>();
    cl::copy(queue, output, C.begin(), C.end());
    return c;
}
