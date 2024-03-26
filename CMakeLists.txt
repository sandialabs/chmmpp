cmake_minimum_required(VERSION 3.14)
project(CHMMPP)

# WEH - This suppresses a cmake warning, but I find this policy confusing.
cmake_policy(SET CMP0144 NEW)

set (CMAKE_CXX_STANDARD 20) #Sets C++ standard to C++20

#
# Global includes
#
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH})
include(Clang-Format)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakePackageConfigHelpers)

#
# Change the default install directory for the sub-builds, as we will
# seldom have access to nor want to install to /usr/local. This can be
# overridden from the command line via cmake -DCMAKE_INSTALL_PREFIX=...
#
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
    CACHE PATH
    "Top level directory for installation"
    FORCE
  )
endif()
set(INCLUDE_INSTALL_DIR "include/" CACHE STRING "Where to install headers relative to prefix")
set(LIBRARY_INSTALL_DIR "lib/" CACHE STRING "Where to install libraries relative to prefix")

#
# Build options
#
option(with_debug "Debug build" OFF)
option(with_verbose "Verbose build" OFF)
option(with_gcov "Add code coverage using gcov" OFF)
option(with_gprof "Profile build" OFF)
option(with_tests "Build tests" OFF)
option(with_boost "Build with boost" ON)
option(coek_home "Installation of Coek" OFF)

if(coek_home)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${coek_home})
else()
    # Assume that Catch2 is installed with coek (for now)
    set(with_tests OFF)
endif()

if (with_gcov)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  else()
    set(with_gcov OFF)
  endif()
endif()
if(with_gcov)
  set(CMAKE_BUILD_TYPE debug)
endif()

if (with_tests)
  find_package(gcovr)
  if (gcovr_FOUND)
    MESSAGE("-- gcovr FOUND: YES")
  else()
    MESSAGE("-- gcovr FOUND: NO")
  endif()
endif()

if(with_verbose)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if(with_debug)
    set(CMAKE_BUILD_TYPE debug)
    if (MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Wall /DDEBUG /permissive-")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -DDEBUG")
    endif()
else()
    if (MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "/permissive-")
    endif()
endif()

#
# Summarize configuration
#
message("-- with_debug: ${with_debug}")
message("-- with_verbose: ${with_verbose}")
message("-- with_gcov: ${with_gcov}")
message("-- with_gprof: ${with_gprof}")
message("-- with_tests: ${with_tests}")
message("-- with_boost: ${with_boost}")
message("-- coek_home: ${coek_home}")

#
# Add subdirectories
#
add_subdirectory(chmmpp)
#add_subdirectory(library)
add_subdirectory(examples)
if (with_tests)
    add_subdirectory(test)
endif()
