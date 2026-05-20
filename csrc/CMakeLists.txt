cmake_minimum_required(VERSION 3.14)
project(LlamaCppUnitTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3 REQUIRED)
find_package(TritonClient REQUIRED)

add_executable(llm_unit_grpc llm_unit_grpc.cc)
target_link_libraries(llm_unit_grpc PRIVATE httpclient grpcclient pthread Catch2::Catch2)

add_executable(llm_unit_kytensor llm_unit_kytensor.cc)
target_link_libraries(llm_unit_kytensor PRIVATE kytensorclient pthread Catch2::Catch2)

option(USE_GCOVR "Use gcovr for coverage" OFF)
if(USE_GCOVR)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov --coverage")
endif()
