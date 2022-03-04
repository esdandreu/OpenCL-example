#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "utils.hpp"

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl2.hpp>
#include <Eigen/Dense>


namespace matmul {

Eigen::MatrixXf opencl(Eigen::MatrixXf& a, Eigen::MatrixXf& b);

}

#endif