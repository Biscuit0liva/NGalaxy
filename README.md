# NGalaxy
N-body problem simulation, using the collision betwen the galaxys Andromeda and the Milky Way. Based on the proyect https://www.evl.uic.edu/sjames/cs525/project2.html

This project was developed in window 11 OS.
## Requirements
- glad
- glfw
- CUDA
- openCL
- CMake
## Execution

In the root, compile with CMake

```bash
cmake -S . -B build
cmake --build build 
```
And execute the simulation with
```bash
./build/Debug/galaxy.exe
```

## TODO

- openCL version

- Experiments
- ### CUDA
  - [ ] Odd vs Even block size value
  - [ ] 2D array vs 1D
  - [ ] Local Memory vs No local memory

- ### openCL
  - [ ] 32 multiple vs no multiple block size value
  - [ ] 2D array vs 1D
  - [ ] Local Memory vs No local memory


