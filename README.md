# Assignment
Make an example. Demonstrate performance speed-up. Demonstrate scalability -
how does performance changes with the number of cores?

- [ ] Explain parallelization model in OpenCL. How can OpenCL communicate with
so many devices?

- [ ] Explain hosts, devices, platforms.

- [ ] How does OpenCL compile kernel code? 

- [ ] Explain OpenCL Installable Client Driver (ICD) and ICD Loader.

- [ ] Explain heterogeneous computing.

- [ ] Identify  a practical problem suitable for solving with OpenCL. Implement
a solution with and without OpenCL support. Demonstrate the performance gain
from using OpenCL solution.

- [ ] Change  the problem size. Observe the difference in performance between
the single threaded and OpenCL solutions as you increase the problem size.

- [ ] Observe the point at which increasing the problem size does not change
the performance gain. Comment why.

# OpenCL
> Open standard for parallel programming of heterogeneous systems

- Kernel

- Host

- Platform: The environment where the program will be executed on

- Device

SDK and Runtime?

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