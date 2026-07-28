cmake_minimum_required(VERSION 3.18)
project(cpp_bullet_full_port_stage06 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_cpp_bullet_full_port_stage06
    src/cpp_line_motion_filter.cpp
    src/pybind_module.cpp
)
target_include_directories(_cpp_bullet_full_port_stage06 PRIVATE include)
