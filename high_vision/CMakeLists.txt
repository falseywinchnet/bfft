cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(high_vision VERSION 0.1.0 LANGUAGES CXX)
endif()

option(HIGH_VISION_BUILD_TESTS "Build High Vision tests" OFF)
option(HIGH_VISION_BUILD_BENCHMARKS "Build the High Vision frame benchmark" OFF)

add_library(high_vision STATIC
  src/high_vision.cpp)
add_library(high_vision::high_vision ALIAS high_vision)
target_include_directories(high_vision PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(high_vision PUBLIC cxx_std_17)
set_target_properties(high_vision PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

# When High Vision is built inside BFFT, use the in-house allocation-free
# TGFD/Meyer engine to produce low-noise registration witnesses. The standalone
# High Vision build retains its dependency-free spatial-pooling fallback.
if(TARGET bfft_static)
  target_link_libraries(high_vision PRIVATE bfft::static)
  target_compile_definitions(high_vision PRIVATE
    HIGH_VISION_WITH_BFFT_MEYER=1)
endif()

if(HIGH_VISION_BUILD_TESTS)
  enable_testing()
  add_executable(high_vision_tests tests/high_vision_tests.cpp)
  target_link_libraries(high_vision_tests PRIVATE high_vision::high_vision)
  target_compile_features(high_vision_tests PRIVATE cxx_std_17)
  add_test(NAME high_vision_tests COMMAND high_vision_tests)
endif()

if(HIGH_VISION_BUILD_BENCHMARKS)
  add_executable(high_vision_benchmark examples/high_vision_benchmark.cpp)
  target_link_libraries(high_vision_benchmark PRIVATE high_vision::high_vision)
  target_compile_features(high_vision_benchmark PRIVATE cxx_std_17)

  add_executable(
    high_vision_entropy_benchmark
    examples/entropy_controller_benchmark.cpp)
  target_link_libraries(
    high_vision_entropy_benchmark PRIVATE high_vision::high_vision)
  target_compile_features(high_vision_entropy_benchmark PRIVATE cxx_std_17)

  add_executable(
    high_vision_night_dynamic_benchmark
    examples/night_dynamic_benchmark.cpp)
  target_link_libraries(
    high_vision_night_dynamic_benchmark PRIVATE high_vision::high_vision)
  target_compile_features(
    high_vision_night_dynamic_benchmark PRIVATE cxx_std_17)
endif()
