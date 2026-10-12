cmake_minimum_required(VERSION 3.16)
project(explosive_sync_flash_crash LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(esfc
  src/main.cpp
  src/network_gen.cpp
  src/kuramoto.cpp
  src/stuart_landau.cpp
  src/flash_crash.cpp
  src/order_param.cpp
)

target_include_directories(esfc PRIVATE src include)
target_compile_options(esfc PRIVATE -O2 -Wall -Wextra -Wpedantic)
target_link_libraries(esfc PRIVATE stdc++fs)
