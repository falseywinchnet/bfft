cmake_minimum_required(VERSION 3.16)

project(bfft VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakePackageConfigHelpers)

option(BFFT_BUILD_SHARED "Build the BFFT shared library" ON)
option(BFFT_BUILD_EXAMPLES "Build BFFT example programs" ON)
option(BFFT_BUILD_TESTS "Build BFFT tests" ON)
option(BFFT_BUILD_PROBES "Build optional BFFT comparison probes" ON)
option(BFFT_ENABLE_AUTO_SIMD "Enable host SIMD flags detected by CMake" ON)
option(BFFT_COMPARE_WITH_IPP "Enable Intel IPP in the library comparison probe when found" ON)
option(BFFT_ENABLE_PFFFT_BENCHMARK "Enable PFFFT in the benchmark example when found" ON)

set(BFFT_PUBLIC_HEADERS
  include/bfft/bfft.h
  include/bfft/bfft.hpp)

set(BFFT_SOURCES
  src/bfft.cpp)

set(BFFT_PLATFORM_LIBS)
if(NOT WIN32)
  list(APPEND BFFT_PLATFORM_LIBS m)
endif()

function(bfft_configure_library target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(${target} PUBLIC cxx_std_17)
  set_target_properties(${target} PROPERTIES
    CXX_EXTENSIONS OFF)
  if(BFFT_PLATFORM_LIBS)
    target_link_libraries(${target} PUBLIC ${BFFT_PLATFORM_LIBS})
  endif()
endfunction()

add_library(bfft_static STATIC ${BFFT_SOURCES} ${BFFT_PUBLIC_HEADERS})
add_library(bfft::static ALIAS bfft_static)
bfft_configure_library(bfft_static)
set_target_properties(bfft_static PROPERTIES
  EXPORT_NAME static
  OUTPUT_NAME bfft)

if(BFFT_BUILD_SHARED)
  add_library(bfft_shared SHARED ${BFFT_SOURCES} ${BFFT_PUBLIC_HEADERS})
  add_library(bfft::shared ALIAS bfft_shared)
  bfft_configure_library(bfft_shared)
  set_target_properties(bfft_shared PROPERTIES
    EXPORT_NAME shared
    OUTPUT_NAME bfft)

  if(WIN32)
    set_target_properties(bfft_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set_target_properties(bfft_static PROPERTIES OUTPUT_NAME bfft_static)
  endif()
endif()

if(BFFT_ENABLE_AUTO_SIMD
    AND NOT MSVC
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
  check_cxx_compiler_flag("-mavx2 -mfma" BFFT_COMPILER_SUPPORTS_AVX2_FMA)
  if(BFFT_COMPILER_SUPPORTS_AVX2_FMA)
    target_compile_options(bfft_static PRIVATE -mavx2 -mfma)
    if(TARGET bfft_shared)
      target_compile_options(bfft_shared PRIVATE -mavx2 -mfma)
    endif()
  endif()
endif()

function(bfft_add_cpp_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE bfft_static)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES
    CXX_EXTENSIONS OFF)
endfunction()

function(bfft_add_c_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE bfft_static)
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

if(BFFT_BUILD_EXAMPLES)
  bfft_add_cpp_executable(benchmark examples/benchmark.cpp)
  if(CMAKE_DL_LIBS)
    target_link_libraries(benchmark PRIVATE ${CMAKE_DL_LIBS})
  endif()

  if(APPLE)
    bfft_add_cpp_executable(apple_benchmark examples/apple_benchmark.cpp)
    find_library(BFFT_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(apple_benchmark PRIVATE ${BFFT_ACCELERATE_FRAMEWORK})
    if(CMAKE_DL_LIBS)
      target_link_libraries(apple_benchmark PRIVATE ${CMAKE_DL_LIBS})
    endif()
  endif()

  bfft_add_cpp_executable(locality_probe examples/locality_probe.cpp)
  bfft_add_cpp_executable(cpp_api_demo examples/cpp_api_demo.cpp)
  bfft_add_c_executable(c_api_demo examples/c_api_demo.c)

  if(BFFT_ENABLE_PFFFT_BENCHMARK)
    find_path(BFFT_PFFFT_INCLUDE_DIR pffft.h)
    find_library(BFFT_PFFFT_LIBRARY NAMES pffft)

    if(BFFT_PFFFT_INCLUDE_DIR AND BFFT_PFFFT_LIBRARY)
      target_compile_definitions(benchmark PRIVATE BFFT_WITH_PFFFT=1)
      target_include_directories(benchmark PRIVATE ${BFFT_PFFFT_INCLUDE_DIR})
      target_link_libraries(benchmark PRIVATE ${BFFT_PFFFT_LIBRARY})
      message(STATUS "PFFFT benchmark comparison enabled")
    else()
      message(STATUS "PFFFT benchmark comparison not enabled; pffft.h or libpffft was not found")
    endif()
  endif()
endif()

if(BFFT_BUILD_TESTS)
  enable_testing()

  bfft_add_cpp_executable(correctness tests/correctness.cpp)
  bfft_add_c_executable(api_c tests/api_c.c)

  add_test(NAME correctness COMMAND correctness)
  add_test(NAME api_c COMMAND api_c)
endif()

if(BFFT_BUILD_PROBES)
  set(BFFT_DL_LIBS "")
  if(CMAKE_DL_LIBS)
    list(APPEND BFFT_DL_LIBS ${CMAKE_DL_LIBS})
  endif()

  set(BFFT_PROBE_SOURCES
    tests/bfft_fftw_support_probe.cpp
    tests/bfft_invariant_support_probe.cpp
    tests/bfft_fftw_sfdr_probe.cpp
    tests/bfft_fftw_sfdr_bh7_probe.cpp
    tests/bfft_library_compare_probe.cpp)

  foreach(probe_source IN LISTS BFFT_PROBE_SOURCES)
    get_filename_component(probe_target ${probe_source} NAME_WE)
    bfft_add_cpp_executable(${probe_target} ${probe_source})
    target_link_libraries(${probe_target} PRIVATE ${BFFT_DL_LIBS})
  endforeach()

  if(BFFT_COMPARE_WITH_IPP)
    find_path(BFFT_IPP_INCLUDE_DIR ipps.h
      HINTS ENV IPPROOT
      PATH_SUFFIXES include)

    find_library(BFFT_IPPS_LIBRARY NAMES ipps
      HINTS ENV IPPROOT
      PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)

    find_library(BFFT_IPPCORE_LIBRARY NAMES ippcore
      HINTS ENV IPPROOT
      PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)

    find_library(BFFT_IPPVM_LIBRARY NAMES ippvm
      HINTS ENV IPPROOT
      PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)

    if(BFFT_IPP_INCLUDE_DIR AND BFFT_IPPS_LIBRARY AND BFFT_IPPCORE_LIBRARY AND BFFT_IPPVM_LIBRARY)
      target_compile_definitions(bfft_library_compare_probe PRIVATE BFFT_COMPARE_WITH_IPP=1)
      target_include_directories(bfft_library_compare_probe PRIVATE ${BFFT_IPP_INCLUDE_DIR})
      target_link_libraries(bfft_library_compare_probe PRIVATE
        ${BFFT_IPPS_LIBRARY}
        ${BFFT_IPPVM_LIBRARY}
        ${BFFT_IPPCORE_LIBRARY})
      message(STATUS "Intel IPP comparison enabled for bfft_library_compare_probe")
    else()
      message(STATUS "Intel IPP comparison not enabled; ipps.h, ipps, ippvm, or ippcore was not found")
    endif()
  endif()
endif()

set(BFFT_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/bfft)
set(BFFT_PC_FILE ${CMAKE_CURRENT_BINARY_DIR}/bfft.pc)

configure_file(pkgconfig/bfft.pc.in ${BFFT_PC_FILE} @ONLY)

configure_package_config_file(
  cmake/BFFTConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BFFTConfig.cmake
  INSTALL_DESTINATION ${BFFT_INSTALL_CMAKEDIR})

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/BFFTConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

set(BFFT_INSTALL_TARGETS bfft_static)
if(TARGET bfft_shared)
  list(APPEND BFFT_INSTALL_TARGETS bfft_shared)
endif()

install(TARGETS ${BFFT_INSTALL_TARGETS}
  EXPORT BFFTTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${BFFT_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bfft)
install(FILES ${BFFT_PC_FILE} DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/BFFTConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/BFFTConfigVersion.cmake
  DESTINATION ${BFFT_INSTALL_CMAKEDIR})

install(EXPORT BFFTTargets
  NAMESPACE bfft::
  DESTINATION ${BFFT_INSTALL_CMAKEDIR})
