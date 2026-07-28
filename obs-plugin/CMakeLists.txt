cmake_minimum_required(VERSION 3.20)

project(bfft-obs VERSION 0.4.0 LANGUAGES C CXX)

include(FetchContent)

set(OBS_SOURCE_DIR "" CACHE PATH
    "OBS Studio source tree (must contain libobs/obs-module.h)")
set(OBS_APP "/Applications/OBS.app" CACHE PATH
    "Installed OBS application used for linking on macOS")
set(OBS_WINDOWS_LIBRARY "" CACHE FILEPATH
    "Windows x64 OBS import library (obs.lib or libobs.dll.a)")

if(NOT EXISTS "${OBS_SOURCE_DIR}/libobs/obs-module.h")
  message(FATAL_ERROR
    "Set OBS_SOURCE_DIR to an OBS Studio source checkout matching your OBS version")
endif()

configure_file(obsconfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/obsconfig.h" @ONLY)

FetchContent_Declare(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG v0.8.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(simde)

set(BFFT_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BFFT_BUILD_PROBES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." bfft-core EXCLUDE_FROM_ALL)

add_library(bfft-cartoon MODULE
  bfft-cartoon-filter.cpp
  "${CMAKE_CURRENT_LIST_DIR}/../high_vision/obs/high-vision-obs-filter.cpp")
target_compile_features(bfft-cartoon PRIVATE cxx_std_17)
target_include_directories(bfft-cartoon PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${simde_SOURCE_DIR}"
  "${OBS_SOURCE_DIR}/libobs"
  "${OBS_SOURCE_DIR}/libobs/util")
target_link_libraries(bfft-cartoon PRIVATE
  bfft::static
  high_vision::high_vision)

if(APPLE)
  target_link_libraries(bfft-cartoon PRIVATE
    "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")
  set_target_properties(bfft-cartoon PROPERTIES
    PREFIX ""
    SUFFIX ""
    BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
  configure_file(Info.plist.in
    "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET bfft-cartoon POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/bfft-cartoon.plugin/Contents/MacOS"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "$<TARGET_FILE:bfft-cartoon>"
      "${CMAKE_CURRENT_BINARY_DIR}/bfft-cartoon.plugin/Contents/MacOS/bfft-cartoon"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
      "${CMAKE_CURRENT_BINARY_DIR}/bfft-cartoon.plugin/Contents/Info.plist"
    COMMAND /usr/bin/codesign --force --sign -
      "${CMAKE_CURRENT_BINARY_DIR}/bfft-cartoon.plugin"
    VERBATIM)
elseif(UNIX)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBOBS REQUIRED IMPORTED_TARGET libobs)
  target_link_libraries(bfft-cartoon PRIVATE PkgConfig::LIBOBS)
  set_target_properties(bfft-cartoon PROPERTIES PREFIX "")
elseif(WIN32)
  if(NOT EXISTS "${OBS_WINDOWS_LIBRARY}")
    message(FATAL_ERROR
      "Set OBS_WINDOWS_LIBRARY to the x64 OBS import library")
  endif()
  target_link_libraries(bfft-cartoon PRIVATE "${OBS_WINDOWS_LIBRARY}")
  set_target_properties(bfft-cartoon PROPERTIES
    PREFIX ""
    SUFFIX ".dll")
  if(MINGW)
    # Produce a self-contained plugin; obs.dll and Windows system DLLs remain
    # the only dynamic runtime dependencies.
    target_link_options(bfft-cartoon PRIVATE
      -static -static-libgcc -static-libstdc++)
  endif()
  add_custom_command(TARGET bfft-cartoon POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/package/bfft-cartoon/obs-plugins/64bit"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "$<TARGET_FILE:bfft-cartoon>"
      "${CMAKE_CURRENT_BINARY_DIR}/package/bfft-cartoon/obs-plugins/64bit/bfft-cartoon.dll"
    VERBATIM)
else()
  message(FATAL_ERROR "Unsupported platform")
endif()
