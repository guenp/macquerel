cmake_minimum_required(VERSION 3.20)
project(macquerel VERSION 0.1 LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Development)

add_subdirectory(src)
add_subdirectory(tests)