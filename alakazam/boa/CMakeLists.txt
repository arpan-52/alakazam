cmake_minimum_required(VERSION 3.16)
project(boa CXX)

# Find Kokkos and KokkosKernels.
# Users pass -DKokkos_ROOT=... -DKokkosKernels_ROOT=... at configure time.
# No hardcoded paths — works with any Kokkos installation.
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

add_subdirectory(cpp)
add_subdirectory(tests)

# Python bindings are optional — only built if pybind11 is found.
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    add_subdirectory(bindings)
else()
    message(STATUS "pybind11 not found — skipping Python bindings")
endif()
