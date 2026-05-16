# Tests for the boa solver library.
# Uses CTest — run with: ctest --test-dir build

enable_testing()

# Helper: add a test executable that links to libboa and registers with CTest.
function(boa_add_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE boa)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

boa_add_test(test_types)
boa_add_test(test_csr_pattern)
boa_add_test(test_residual)
boa_add_test(test_jacobian)
boa_add_test(test_linear_solvers)
boa_add_test(test_gain_solver)
boa_add_test(test_delay_solver)
boa_add_test(test_leakage_solver)
boa_add_test(test_cross_delay_solver)
boa_add_test(test_cross_phase_solver)
boa_add_test(test_ref_antenna)
