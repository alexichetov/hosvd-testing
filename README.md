## Dependencies
1. LibTorch 2.9.1 CPU-only
2. CMake
3. Python3: with pandas and matplotlib

## How to build
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<PATH TO PROJECT>/libtorch -DCMAKE_BUILD_TYPE=Release ..
make
