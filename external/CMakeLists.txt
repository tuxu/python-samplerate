include(FetchContent)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8 # 3.0.1
)

FetchContent_MakeAvailable(pybind11)

# libsamplerate
set(BUILD_TESTING OFF CACHE BOOL "Disable libsamplerate test build")

FetchContent_Declare(
  libsamplerate
  GIT_REPOSITORY https://github.com/libsndfile/libsamplerate
  GIT_TAG c96f5e3de9c4488f4e6c97f59f5245f22fda22f7 # 0.2.2
)
set(LIBSAMPLERATE_VERSION 0.2.2 CACHE STRING PUBLIC)  # <-- update libsamplerate version here

FetchContent_MakeAvailable(libsamplerate)
