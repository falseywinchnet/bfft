cmake_minimum_required(VERSION 3.20)

project(convstar-support-obs VERSION 0.5.0 LANGUAGES C CXX)

set(OBS_SOURCE_DIR "" CACHE PATH
    "OBS Studio source tree (must contain libobs/obs-module.h)")
set(OBS_APP "/Applications/OBS.app" CACHE PATH
    "Installed OBS application used for linking on macOS")
set(SIMDE_SOURCE_DIR "" CACHE PATH
    "SIMDe source tree required by OBS public headers")

if(NOT EXISTS "${OBS_SOURCE_DIR}/libobs/obs-module.h")
  message(FATAL_ERROR
    "Set OBS_SOURCE_DIR to an OBS Studio source checkout matching your OBS version")
endif()
if(NOT EXISTS "${SIMDE_SOURCE_DIR}/simde/x86/sse2.h")
  message(FATAL_ERROR "Set SIMDE_SOURCE_DIR to a SIMDe source checkout")
endif()

configure_file(obsconfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/obsconfig.h" @ONLY)

add_library(convstar-support MODULE
  convstar-support-filter.cpp
  dip-comb-transport.cpp)
target_compile_features(convstar-support PRIVATE cxx_std_17)
target_include_directories(convstar-support PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${SIMDE_SOURCE_DIR}"
  "${OBS_SOURCE_DIR}/libobs"
  "${OBS_SOURCE_DIR}/libobs/util")

if(APPLE)
  enable_language(OBJCXX)
  target_link_libraries(convstar-support PRIVATE
    "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")
  set_target_properties(convstar-support PROPERTIES
    PREFIX ""
    SUFFIX ""
    BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
  configure_file(Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET convstar-support POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/convstar-support.plugin/Contents/MacOS"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "$<TARGET_FILE:convstar-support>"
      "${CMAKE_CURRENT_BINARY_DIR}/convstar-support.plugin/Contents/MacOS/convstar-support"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
      "${CMAKE_CURRENT_BINARY_DIR}/convstar-support.plugin/Contents/Info.plist"
    COMMAND /usr/bin/codesign --force --sign -
      "${CMAKE_CURRENT_BINARY_DIR}/convstar-support.plugin"
    VERBATIM)

  add_executable(convstar-support-smoke convstar-support-smoke.mm)
  target_compile_features(convstar-support-smoke PRIVATE cxx_std_17)
  target_include_directories(convstar-support-smoke PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${SIMDE_SOURCE_DIR}"
    "${OBS_SOURCE_DIR}/libobs"
    "${OBS_SOURCE_DIR}/libobs/util")
  target_link_libraries(convstar-support-smoke PRIVATE
    "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs"
    "-framework AppKit")
  set_target_properties(convstar-support-smoke PROPERTIES
    BUILD_RPATH "${OBS_APP}/Contents/Frameworks")

  add_executable(dip-comb-transport-test
    dip-comb-transport-test.cpp
    dip-comb-transport.cpp)
  target_compile_features(dip-comb-transport-test PRIVATE cxx_std_17)
elseif(UNIX)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBOBS REQUIRED IMPORTED_TARGET libobs)
  target_link_libraries(convstar-support PRIVATE PkgConfig::LIBOBS)
  set_target_properties(convstar-support PROPERTIES PREFIX "")
else()
  message(FATAL_ERROR "The DIP-CONV OBS package targets macOS/Linux")
endif()
