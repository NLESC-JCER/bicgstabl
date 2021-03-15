cmake_minimum_required (VERSION 3.10)
project (benchmark CXX)

## Set CMAKE options

#Directory were the external cmake modules are stored
include(FeatureSummary)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules/")
#Make the default build release is this in what a normal user wants
###################################################################
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE
			STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
	set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel )
endif(NOT CMAKE_BUILD_TYPE)


#allow dynamically locating dynamic libraries if cmake and the OS support it
set(CMAKE_MACOSX_RPATH 1)



#Here is the check for CXX14 support : We now use some features of this, so turn it on if possible
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

option(ENABLE_WARNING_FLAGS "Inject more warning flags" OFF)
if(ENABLE_WARNING_FLAGS)
  include(CheckCXXCompilerFlag)
  foreach(_FLAG -Wall -Wextra -Wpedantic -Wshadow -Wconversion) 
    check_cxx_compiler_flag("${_FLAG}" COMPILER_SUPPORTS${_FLAG})
    if(COMPILER_SUPPORTS${_FLAG})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_FLAG}")
    endif()
  endforeach()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcheck")
  endif()
endif()

option(ENABLE_WERROR "Inject -Werror" OFF)
if(ENABLE_WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

option(ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(ENABLE_COVERAGE_BUILD)
    message(STATUS "Enabling coverage build")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()


find_package(OpenMP 4.5 REQUIRED)
set_package_properties(OpenMP PROPERTIES TYPE RECOMMENDED PURPOSE "Used for thread parallelization")

find_package(Eigen3 3.3.0 NO_MODULE REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED PURPOSE "C++ vector data structures")
message(STATUS "Found Eigen3: ${Eigen3_DIR}")


add_subdirectory(include)
add_subdirectory(src)






feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
