# NGalaxy
N-body problem simulation, using the collision betwen the galaxys Andromeda and the Milky Way. Based on the proyect https://www.evl.uic.edu/sjames/cs525/project2.html

This project was developed in window 11 OS and Ubuntu .
## Requirements
- glad
- glfw
- CUDA
- openCL
- CMake 3.24+
## Execution

At the project root, compile with CMake specifying whether to use CUDA (default) or OpenCL.
To do so, run the same commands twice, changing the value of **USE_OPENCL**

### For CUDA (default)

```bash
cmake -S . -B build -DUSE_OPENCL=OFF
cmake --build build 
```
### For OpenCL

```bash
cmake -S . -B build -DUSE_OPENCL=ON
cmake --build build 
```
For the execution, the path changes.
### For CUDA
original version
```bash
./build/cuda/Debug/galaxy_cuda.exe
```
2D mapping version
```bash
./build/cuda/2D/Debug/galaxy_cuda_2D.exe
```
Global memory version
```bash
./build/cuda/global/Debug/galaxy_cuda_global.exe
```
### For OpenCL
original version
```bash
./build/opencl/Debug/galaxy_galaxy_opencl.exe
```
2D mapping version
```bash
./build/opencl/2D/Debug/galaxy_opencl_2D.exe
```
Global memory version
```bash
./build/opencl/global/Debug/galaxy_opencl_global.exe
```

### Note for hybrid GPU systems (Intel + NVIDIA)

On machines with both an integrated Intel GPU and a discrete NVIDIA GPU (Optimus systems), the OpenGL context may default to the Intel GPU, while CUDA runs on the NVIDIA GPU. To enable CUDA–OpenGL interoperability (e.g., cudaGraphicsMapResources), you must force the application to use the NVIDIA GPU. For example:
```bash
prime-run ./build/.../galaxy.exe
```
If prime-run is not available, you can achieve the same effect by using environment variables:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
prime-run ./build/.../galaxy.exe
```

On systems with a single dedicated NVIDIA GPU, no special steps are required.

## TODO

- Experiments
- ### CUDA
  - [x] 32 multiple vs no multiple block size value
  - [x] 2D array vs 1D
  - [x] Local Memory vs No local memory

- ### openCL
  - [x] 32 multiple vs no multiple block size value
  - [x] 2D array vs 1D
  - [x] Local Memory vs No local memory


