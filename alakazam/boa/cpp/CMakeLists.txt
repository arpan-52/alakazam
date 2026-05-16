# The boa library — portable calibration solver using Kokkos.
# Links against Kokkos and KokkosKernels for execution-space-portable
# sparse linear algebra (SpMV, BLAS, etc.).

add_library(boa
    src/csr_pattern.cpp
    src/residual.cpp
    src/jacobian.cpp
    src/linear_solvers.cpp
    src/api.cpp
)

target_include_directories(boa PUBLIC include)

# Kokkos::kokkos provides parallel_for, Views, execution spaces.
# Kokkos::kokkoskernels provides KokkosBlas, KokkosSparse::spmv, CrsMatrix.
target_link_libraries(boa PUBLIC Kokkos::kokkos Kokkos::kokkoskernels)

# Require C++17 (Kokkos 4.x needs it).
target_compile_features(boa PUBLIC cxx_std_17)

# PIC required so libboa.a can be linked into a shared library (pybind11 module).
set_target_properties(boa PROPERTIES POSITION_INDEPENDENT_CODE ON)
