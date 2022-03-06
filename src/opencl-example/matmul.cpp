#include "matmul.hpp"
#include <iostream>

matmul::opencl::opencl(cl::Device& device) : device(device) {
    // if num_threads -> createSubDevices()
    // std::vector<cl::Device> devices;
    // cl_device_partition_property properties = {
    // CL_DEVICE_PARTITION_BY_COUNTS,
    //     1, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };
    // if (device.createSubDevices(&properties, &devices) != CL_SUCCESS) {
    //     std::cout << "Error creating subdevices" << std::endl;
    // }
    cl_int err;
    try {
        context = cl::Context(device, /*properties*/ nullptr,
            /*callback*/ nullptr,
            /*data*/ nullptr, &err);
    } catch (...) {
        std::cout << "Error creating context" << std::endl;
    }
    std::cout << "Hello world " << err << std::endl;
    queue   = cl::CommandQueue(context);
    program = matmul::cl_utils::build_program(context, "matmul");
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
