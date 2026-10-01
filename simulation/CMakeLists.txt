cmake_minimum_required(VERSION 3.18)
project(bhsim LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BHSIM_ENABLE_CUDA "Build the CUDA ray tracer"        OFF)
option(BHSIM_BUILD_TESTS "Build the C++ tests"             ON)
option(BHSIM_PYTHON      "Build the pybind11 Python module" OFF)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Optimization flags (GCC/Clang only; -march=native probed, not assumed)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<CONFIG:Release>:-O3>)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" BHSIM_HAS_MARCH_NATIVE)
  if(BHSIM_HAS_MARCH_NATIVE)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP (required)
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Core library: metrics + integrators, shared by the CLI and the tests
# ---------------------------------------------------------------------------
add_library(bhsim_core
  src/schwarzschild.cpp
  src/kerr.cpp
  src/raytrace.cpp
)
target_include_directories(bhsim_core PUBLIC include)
target_link_libraries(bhsim_core PUBLIC OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# CLI executable: parameters in, trajectories out
# ---------------------------------------------------------------------------
add_executable(bhsim src/main.cpp)
target_link_libraries(bhsim PRIVATE bhsim_core)

# ---------------------------------------------------------------------------
# Optional CUDA ray tracer (off by default so CPU-only machines still build)
# ---------------------------------------------------------------------------
if(BHSIM_ENABLE_CUDA)
  enable_language(CUDA)
  add_executable(bhsim_raytrace cuda/raytrace.cu)
  target_include_directories(bhsim_raytrace PRIVATE include)
  set_target_properties(bhsim_raytrace PROPERTIES CUDA_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# Optional Python bindings (pybind11). Off by default; enable with
#   cmake -S sim -B sim/build -DBHSIM_PYTHON=ON
# pybind11 is fetched automatically, and the built module `_bhsim` is placed
# inside the bhml/ package so `from bhml import _bhsim` works from the repo root.
# ---------------------------------------------------------------------------
if(BHSIM_PYTHON)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(_bhsim
    ../bhml/bindings/bhsim_py.cpp
    src/schwarzschild.cpp
  )
  target_include_directories(_bhsim PRIVATE include)

  set(_bhml_dir ${CMAKE_CURRENT_SOURCE_DIR}/../bhml)
  set_target_properties(_bhsim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY         ${_bhml_dir}
    RUNTIME_OUTPUT_DIRECTORY         ${_bhml_dir}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_bhml_dir}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_bhml_dir}
  )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BHSIM_BUILD_TESTS)
  enable_testing()
  add_executable(test_schwarzschild tests/test_schwarzschild.cpp)
  target_link_libraries(test_schwarzschild PRIVATE bhsim_core)
  add_test(NAME schwarzschild COMMAND test_schwarzschild)

  add_executable(test_kerr tests/test_kerr.cpp)
  target_link_libraries(test_kerr PRIVATE bhsim_core)
  target_compile_definitions(test_kerr PRIVATE
    KERR_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/kerr_ref_a05.csv")
  add_test(NAME kerr COMMAND test_kerr)
endif()
