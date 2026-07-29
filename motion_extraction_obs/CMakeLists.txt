cmake_minimum_required(VERSION 3.20)
project(posy-motion-obs VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG v0.8.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(simde)

set(OBS_SOURCE_DIR "" CACHE PATH "OBS Studio source tree")
set(OBS_APP "/Applications/OBS.app" CACHE PATH "Installed OBS application")

if(NOT EXISTS "${OBS_SOURCE_DIR}/libobs/obs-module.h")
  message(FATAL_ERROR "Set OBS_SOURCE_DIR to an OBS Studio source checkout")
endif()

configure_file(obsconfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/obsconfig.h" @ONLY)

add_library(posy-motion MODULE posy-motion-filter.cpp)
target_compile_features(posy-motion PRIVATE cxx_std_17)
target_include_directories(posy-motion PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${simde_SOURCE_DIR}"
  "${OBS_SOURCE_DIR}/libobs"
  "${OBS_SOURCE_DIR}/libobs/util")

if(APPLE)
  target_link_libraries(posy-motion PRIVATE
    "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")
  set_target_properties(posy-motion PROPERTIES PREFIX "" SUFFIX "")
  configure_file(Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET posy-motion POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/posy-motion.plugin/Contents/MacOS"
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:posy-motion>"
      "${CMAKE_CURRENT_BINARY_DIR}/posy-motion.plugin/Contents/MacOS/posy-motion"
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
      "${CMAKE_CURRENT_BINARY_DIR}/posy-motion.plugin/Contents/Info.plist"
    COMMAND /usr/bin/codesign --force --sign -
      "${CMAKE_CURRENT_BINARY_DIR}/posy-motion.plugin")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBOBS REQUIRED IMPORTED_TARGET libobs)
  target_link_libraries(posy-motion PRIVATE PkgConfig::LIBOBS)
  set_target_properties(posy-motion PROPERTIES PREFIX "")
endif()
