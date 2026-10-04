cmake_minimum_required(VERSION 3.16)
project(mosaic_fx VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mosaic_core STATIC src/mosaic.cpp)
target_include_directories(mosaic_core PUBLIC include)
target_compile_options(mosaic_core PRIVATE -O3)

# Posterizer Mark IV supplies the tile palette.
add_library(mosaic_rvfx STATIC ../realtime_vector_fx/src/engine.cpp)
target_include_directories(mosaic_rvfx PUBLIC ../realtime_vector_fx/include)
target_compile_options(mosaic_rvfx PRIVATE -O3 -ffast-math)

add_executable(mosaic_tests tests/test_mosaic.cpp)
target_link_libraries(mosaic_tests PRIVATE mosaic_core)

option(MOSAIC_WITH_BFFT "Build tools that take the Meyer cartoon from BFFT" ON)
if(MOSAIC_WITH_BFFT)
  set(BFFT_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(BFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BFFT_BUILD_PROBES OFF CACHE BOOL "" FORCE)
  set(BFFT_BUILD_HIGH_VISION OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." bfft-core EXCLUDE_FROM_ALL)
  add_library(mosaic_cartoon STATIC src/cartoon.cpp)
  target_include_directories(mosaic_cartoon PUBLIC include)
  target_compile_options(mosaic_cartoon PRIVATE -O3)
  target_link_libraries(mosaic_cartoon PUBLIC bfft::static)
  add_executable(mosaic_still tools/mosaic_still.cpp)
  target_link_libraries(mosaic_still PRIVATE mosaic_core mosaic_rvfx mosaic_cartoon)
endif()

option(MOSAIC_BUILD_OBS "Build the OBS plugin and its libobs smoke harness" OFF)
if(MOSAIC_BUILD_OBS)
  if(NOT MOSAIC_WITH_BFFT)
    message(FATAL_ERROR "MOSAIC_BUILD_OBS requires MOSAIC_WITH_BFFT")
  endif()
  set(OBS_SOURCE_DIR "/tmp/obs-studio-32.2.1" CACHE PATH "OBS source matching the target OBS runtime")
  set(OBS_CONFIG_INCLUDE_DIR "/tmp" CACHE PATH "Directory containing obsconfig.h")
  set(SIMDE_INCLUDE_DIR "/tmp/simde-0.8.2" CACHE PATH "Directory containing simde/")
  set(OBS_APP "/Applications/OBS.app" CACHE PATH "OBS application used for linking")
  set(OBS_INCLUDES "${OBS_CONFIG_INCLUDE_DIR}" "${OBS_SOURCE_DIR}/libobs" "${OBS_SOURCE_DIR}/libobs/util" "${SIMDE_INCLUDE_DIR}")
  set(LIBOBS "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")

  add_library(mosaic-fx MODULE obs/mosaic-filter.cpp)
  target_include_directories(mosaic-fx PRIVATE ${OBS_INCLUDES})
  target_compile_options(mosaic-fx PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
  target_link_libraries(mosaic-fx PRIVATE mosaic_core mosaic_rvfx mosaic_cartoon "${LIBOBS}")
  set_target_properties(mosaic-fx PROPERTIES PREFIX "" SUFFIX "" BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
  configure_file(obs/Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  set(BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/mosaic-fx.plugin")
  add_custom_command(TARGET mosaic-fx POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${BUNDLE}/Contents/MacOS"
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:mosaic-fx>" "${BUNDLE}/Contents/MacOS/mosaic-fx"
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" "${BUNDLE}/Contents/Info.plist"
    COMMAND /usr/bin/codesign --force --sign - "${BUNDLE}"
    VERBATIM)

  enable_language(OBJCXX)
  add_executable(mosaic-fx-smoke tools/mosaic_obs_smoke.mm)
  target_include_directories(mosaic-fx-smoke PRIVATE ${OBS_INCLUDES})
  target_link_libraries(mosaic-fx-smoke PRIVATE mosaic_core "${LIBOBS}" "-framework AppKit")
  set_target_properties(mosaic-fx-smoke PROPERTIES BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
endif()

enable_testing()
add_test(NAME mosaic_tests COMMAND mosaic_tests)
