This example is part of an assignment for the course Concurrent Computing in
Robotics from JEMARO.

## OpenCL - First Steps
Head to the [report
document](https://gitlab.com/jemaro/concurrent-computing-in-robotics/opencl-essay/-/raw/main/Essay_Group_4_Assignment_2_Gimenez_Andreu_OpenCL.pdf)
for detailed information about the first steps in OpenCL programming.
## Setup
### Install the requirements
Install [vcpkg](https://github.com/microsoft/vcpkg) requirements with the
addition of `cmake` and Python. It could be summarized as:
- [git](https://git-scm.com/downloads)
- Build tools ([Visual
  Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio)
  on Windows or `gcc` on Linux for example)
- [cmake](#cmake)

If running on a clean linux environment (like a container or Windows Subsystem
for Linux) you will need to install some additional tools as it is stated in
`vcpkg`.
```
sudo apt-get install build-essential curl zip unzip tar pkg-config libssl-dev python3-dev
```
#### CMake
Follow the [official instructions](https://cmake.org/install/).

### Clone this repository with `vcpkg`

Cone this repository with `vcpkg` as a submodule and navigate into it.
```
git clone --recursive git@github.com:esdandreu/OpenCL-example.git
cd OpenCL-example
```

Bootstrap `vcpkg` in Windows. Make sure you have [installed the
prerequisites](https://github.com/microsoft/vcpkg).
```
.\vcpkg\bootstrap-vcpkg.bat
```

Or in Linux/MacOS. Make sure you have [installed developer
tools](https://github.com/microsoft/vcpkg)
```
./vcpkg/bootstrap-vcpkg.sh
```

## Building

### Build locally with CMake
Navigate to the root of the repository and create a build directory.
```
mkdir build
```

Configure `cmake` to use `vcpkg`.
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$pwd/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Build the project.
```
cmake --build build
```

## Testing

### Test the C++ library with Google Test

```
ctest --test-dir build
```

## Benchmark

### Benchmark the C++ library with Google Benchmark

Run the compiled executable
```
benchmark_opencl-example.exe --benchmark_out=benchmark/results.json
```