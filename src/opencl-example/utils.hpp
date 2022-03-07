#ifndef UTILS_HPP
#define UTILS_HPP

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 210
#endif

#include <CL/cl2.hpp>
#include <filesystem>
#include <fstream>

namespace matmul {
namespace cl_utils {

static std::filesystem::path get_program_path(const std::string& name) {
    // The program must be located in the same directory as this source file
    return std::filesystem::path(__FILE__).parent_path() / (name + ".cl");
}

static cl::Program
build_program(const cl::Context& context, const std::string& name) {
    // ! Can fail if the file does not exist
    std::ifstream stream(get_program_path(name));

    std::string source(std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
    return cl::Program(context, source, /* build */ true);
}

static inline std::vector<cl::Device> get_all_devices() {
    std::vector<cl::Device> all_devices;
    std::vector<cl::Platform> platforms;
    cl_int err = cl::Platform::get(&platforms);
    for (auto& p : platforms) {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        // for (auto& d : devices) {
        //     cl::Device device_copy(d);
        //     all_devices.push_back(device_copy);
        // }
        all_devices.insert(all_devices.end(),
            std::make_move_iterator(devices.begin()),
            std::make_move_iterator(devices.end()));
    }
    return all_devices;
}

} // namespace cl_utils
} // namespace matmul

#endif