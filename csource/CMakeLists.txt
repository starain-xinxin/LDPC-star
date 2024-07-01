cmake_minimum_required(VERSION 3.14)
project(LLR_BP_CUDA LANGUAGES CXX CUDA)

# 设置 CUDA 相关配置
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
set(CMAKE_CUDA_STANDARD 11)

# 设置 CUDA 体系结构
set(CMAKE_CUDA_ARCHITECTURES 75) # 75 是指 CUDA Compute Capability 7.5，适用于较新的 GPU

# 设置 Python 相关配置
set(PYTHON_EXECUTABLE "/home/yuanxinyu/anaconda3/envs/LDPC/bin/python")
set(PYTHON_INCLUDE_DIR "/home/yuanxinyu/anaconda3/envs/LDPC/include/python3.8")

# 添加 pybind11 子目录
add_subdirectory(extern/pybind11)

# 添加 CUDA 源文件并创建库
pybind11_add_module(LLR_BP_CUDA LLR_BP_CUDA.cu)
set_target_properties(LLR_BP_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 链接 pybind11 库
target_link_libraries(LLR_BP_CUDA PRIVATE pybind11::module)

# 设置 LLR_BP_CUDA 的属性
set_target_properties(LLR_BP_CUDA PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
