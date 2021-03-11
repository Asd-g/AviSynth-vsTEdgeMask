cmake_minimum_required(VERSION 3.16)

project(libvstedgemask LANGUAGES CXX)

find_package (Git)

if (GIT_FOUND)
    execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        OUTPUT_VARIABLE ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    message (STATUS "GIT not found")
endif ()

add_library(vstedgemask SHARED
    src/vsTEdgeMask.cpp
    src/vsTEdgeMask_SSE2.cpp
)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")

set_source_files_properties(src/vsTEdgeMask_SSE2.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse2")

target_include_directories(vstedgemask PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include/avisynth
)

set_target_properties(vstedgemask PROPERTIES OUTPUT_NAME "vstedgemask.${ver}")

target_compile_features(vstedgemask PRIVATE cxx_std_11)

include(GNUInstallDirs)

INSTALL(TARGETS vstedgemask LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")
        
# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
