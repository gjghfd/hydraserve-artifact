cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

execute_process(COMMAND ls /.dockerenv OUTPUT_VARIABLE _DOCKER_ENV)
# if (NOT _DOCKER_ENV)
#   # use conda environment
#   link_directories($ENV{CONDA_PREFIX}/lib)
#   include_directories($ENV{CONDA_PREFIX}/include)
#   set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

#   set(CMAKE_CUDA_COMPILER $ENV{CONDA_PREFIX}/bin/nvcc)
#   set(CUDAToolkit_ROOT $ENV{CONDA_PREFIX})
# else()
#   message(STATUS "Running in docker, not setting conda environment")
# endif()

project(FastLoad LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.8 REQUIRED)

# gcc >= 8 is required, we do not support other compilers
if ((NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0))
    message(FATAL_ERROR "GCC 8.0 or higher is required")
endif()
# Add filesystem library for gcc < 9
link_libraries( "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>" )

# Set up C++ standard
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Switch between release mode and debug mode
# The user can use `-DBUILD_MODE=DEBUG` or `-DBUILD_MODE=RELEASE` to
# choose the build mode.
# If no option is provided, default to debug mode
if (BUILD_MODE)
  string(TOUPPER ${BUILD_MODE} BUILD_MODE)
  if (BUILD_MODE STREQUAL "DEBUG")
    set(DEBUG ON)
  elseif (BUILD_MODE STREQUAL "RELEASE")
    set(RELEASE ON)
  else()
    message(FATAL_ERROR "Unknown build mode: ${BUILD_MODE}")
  endif()
else()
  message("No build type selected, defaulting to RELEASE mode")
  message("Use -DBUILD_MODE=DEBUG or -DBUILD_MODE=RELEASE to specify build type")
  set(RELEASE ON)
endif()

# Set up C++ flag and CUDA flag
if (DEBUG)
  message("Building in debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DDEBUG")
else()
  message("Building in release mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --prec-div=false")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DRELEASE")
endif()

# Set up MPI and NCCL for multi-GPU communication
message("Building with MPI and NCCL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(MKL_MPI "openmpi")
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

# Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/csrc
)
set(COMMON_LIB_DIRS "")
set(COMMON_LIBS CUDA::cudart)

# Add Python into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
set(PYTHON_PATH "python3" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
print(sysconfig.get_paths()['include']);"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Python config Error.")
endif()
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})


# Add LibTorch into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
print(os.path.dirname(torch.__file__), end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)
list(APPEND COMMON_HEADER_DIRS "${TORCH_INCLUDE_DIRS}")
list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")


# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})


# Should turn off CXX11 ABI if pytorch is built with CXX11 ABI off
execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Let all executable targets go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add subdirectories
add_subdirectory(src)
