# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

# We want to use the Python (anaconda) prefix
if (WIN32)
  get_filename_component(PYTHON_PREFIX ${PYTHON_EXECUTABLE} DIRECTORY)
else()
  exec_program("which python | sed 's:/bin/python::'"
    OUTPUT_VARIABLE PYTHON_PREFIX
    RETURN_VALUE PYTHON_NOT_FOUND
    )
  if(PYTHON_NOT_FOUND)
      message(FATAL_ERROR "Python prefix not found")
  endif()
endif()

message(STATUS "Found Python prefix ${PYTHON_PREFIX}")
set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
message("The cmake module path ${CMAKE_MODULE_PATH}")
message("The current source dir ${CMAKE_CURRENT_SOURCE_DIR}")


project(python-samplerate)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonInterp)

# This goes after, since it uses PythonInterp as hint
find_package(PythonLibs)

# Use just one of these:
if (MSVC)
  set(CMAKE_CXX_STANDARD 14)
else()
  # GCC/clang:
  set(CMAKE_CXX_STANDARD 14)
endif()

# adds the external dependencies
add_subdirectory(external)

pybind11_add_module(python-samplerate src/python-samplerate.cpp)

target_include_directories(python-samplerate PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(python-samplerate PRIVATE ${NUMPY_INCLUDE_DIRS})
target_include_directories(python-samplerate PRIVATE ./external/libsamplerate/include)

if(MSVC)
    target_compile_options(python-samplerate PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(python-samplerate PRIVATE -std=c++14 -O3 -Wall -Wextra)
endif()

add_definitions(-DVERBOSE=1)

### Final target setup
set_target_properties(
    python-samplerate
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "samplerate"
        LINKER_LANGUAGE C
    )

target_link_libraries(python-samplerate PUBLIC samplerate)
