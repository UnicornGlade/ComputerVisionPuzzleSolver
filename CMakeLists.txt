cmake_minimum_required(VERSION 3.16)
project(sobel_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(third_party/stb)
add_subdirectory(libs/base)
add_subdirectory(libs/images)
add_subdirectory(src/sobel)
