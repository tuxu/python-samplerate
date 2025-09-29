# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.15)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

message(STATUS "Found Python prefix ${PYTHON_PREFIX}")
list(PREPEND CMAKE_PREFIX_PATH "${PYTHON_PREFIX}")

project(python-samplerate)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0148 NEW)


option(USE_SYSTEM_LIBRARIES "Use system libsamplerate and pybind instead of building separate versions" OFF)
if(USE_SYSTEM_LIBRARIES)
	find_package(pybind11 REQUIRED)
else()
	# adds the external dependencies
	add_subdirectory(external)
endif()

pybind11_add_module(python-samplerate src/samplerate.cpp)

if(USE_SYSTEM_LIBRARIES)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(SAMPLERATE REQUIRED IMPORTED_TARGET GLOBAL samplerate)
	target_link_libraries(python-samplerate PUBLIC PkgConfig::SAMPLERATE)
else()
	target_include_directories(python-samplerate PRIVATE ./external/libsamplerate/include)
	target_link_libraries(python-samplerate PUBLIC samplerate)
endif()

if(MSVC)
    target_compile_options(python-samplerate PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(python-samplerate PRIVATE -std=c++14 -O3 -Wall -Wextra -fPIC)
endif()

### stick the package and libsamplerate version into the module
target_compile_definitions(python-samplerate
    PUBLIC LIBSAMPLERATE_VERSION="${LIBSAMPLERATE_VERSION}"
    PRIVATE $<$<BOOL:${PACKAGE_VERSION_INFO}>:VERSION_INFO="${PACKAGE_VERSION_INFO}">
)

### Final target setup
set_target_properties(
    python-samplerate
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "samplerate"
        LINKER_LANGUAGE C
    )

