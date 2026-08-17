cmake_minimum_required(VERSION 3.20)
project(realtime-vector-fx VERSION 0.1.0 LANGUAGES CXX)

option(RVFX_BUILD_OBS "Build the OBS Studio filter" OFF)
set(OBS_SOURCE_DIR "" CACHE PATH "OBS Studio source tree")
set(OBS_CONFIG_INCLUDE_DIR "" CACHE PATH "Directory containing generated obsconfig.h")
set(SIMDE_INCLUDE_DIR "" CACHE PATH "Directory containing simde/")
set(OBS_APP "/Applications/OBS.app" CACHE PATH "Installed OBS application")

add_library(rvfx_core STATIC src/engine.cpp)
target_compile_features(rvfx_core PUBLIC cxx_std_17)
target_include_directories(rvfx_core PUBLIC include)
target_compile_options(rvfx_core PRIVATE -O3 -ffast-math)
set_target_properties(rvfx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(rvfx_tests tests/test_engine.cpp)
target_link_libraries(rvfx_tests PRIVATE rvfx_core)

add_executable(rvfx_benchmark tools/benchmark.cpp)
target_link_libraries(rvfx_benchmark PRIVATE rvfx_core)

add_executable(rvfx_demo tools/demo.cpp)
target_link_libraries(rvfx_demo PRIVATE rvfx_core)

enable_testing()
add_test(NAME rvfx_core COMMAND rvfx_tests)

if(RVFX_BUILD_OBS)
  if(NOT EXISTS "${OBS_SOURCE_DIR}/libobs/obs-module.h")
    message(FATAL_ERROR "Set OBS_SOURCE_DIR to an OBS Studio source checkout")
  endif()
  add_library(realtime-vector-fx MODULE obs/plugin-main.cpp obs/gpu-filter.cpp)
  target_compile_features(realtime-vector-fx PRIVATE cxx_std_17)
  target_include_directories(realtime-vector-fx PRIVATE
    include "${OBS_CONFIG_INCLUDE_DIR}" "${OBS_SOURCE_DIR}/libobs"
    "${OBS_SOURCE_DIR}/libobs/util" "${SIMDE_INCLUDE_DIR}")
  target_link_libraries(realtime-vector-fx PRIVATE rvfx_core)
  if(APPLE)
    target_link_libraries(realtime-vector-fx PRIVATE
      "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")
    set_target_properties(realtime-vector-fx PROPERTIES
      PREFIX "" SUFFIX "" BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
    configure_file(obs/Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
    add_custom_command(TARGET realtime-vector-fx POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/realtime-vector-fx.plugin/Contents/MacOS"
      COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:realtime-vector-fx>"
        "${CMAKE_CURRENT_BINARY_DIR}/realtime-vector-fx.plugin/Contents/MacOS/realtime-vector-fx"
      COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
        "${CMAKE_CURRENT_BINARY_DIR}/realtime-vector-fx.plugin/Contents/Info.plist"
      COMMAND /usr/bin/codesign --force --sign -
        "${CMAKE_CURRENT_BINARY_DIR}/realtime-vector-fx.plugin"
      VERBATIM)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBOBS REQUIRED IMPORTED_TARGET libobs)
    target_link_libraries(realtime-vector-fx PRIVATE PkgConfig::LIBOBS)
    set_target_properties(realtime-vector-fx PROPERTIES PREFIX "")
  endif()
endif()
