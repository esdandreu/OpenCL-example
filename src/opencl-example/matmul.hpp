#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "utils.hpp"

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl2.hpp>
#include <Eigen/Dense>


namespace matmul {

class opencl {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    public:
    cl::Device device;

    opencl(cl::Device& device = cl::Device::getDefault(), int num_units = NULL);
    Eigen::MatrixXf operator()(Eigen::MatrixXf& a, Eigen::MatrixXf& b);
};


} // namespace matmul

#endif