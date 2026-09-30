cmake_minimum_required(VERSION 3.18)
project(bhsim LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BHSIM_ENABLE_CUDA "Build the CUDA ray tracer" OFF)
option(BHSIM_BUILD_TESTS "Build the C++ tests"       ON)

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
# Tests
# ---------------------------------------------------------------------------
if(BHSIM_BUILD_TESTS)
  enable_testing()
  add_executable(test_schwarzschild tests/test_schwarzschild.cpp)
  target_link_libraries(test_schwarzschild PRIVATE bhsim_core)
  add_test(NAME schwarzschild COMMAND test_schwarzschild)
endif()
