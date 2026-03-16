cmake_minimum_required(VERSION 3.20)
project(evo_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(evo_core
    cpp/agent.cpp
    cpp/simulation_core.cpp
    cpp/evo_core.cpp
)

target_include_directories(evo_core PRIVATE cpp)

target_compile_options(evo_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>
)
