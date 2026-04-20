cmake_minimum_required(VERSION 3.10)
project(morphology_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(morphology_cpp
    morphology.cpp
    bindings.cpp
)

target_include_directories(morphology_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
