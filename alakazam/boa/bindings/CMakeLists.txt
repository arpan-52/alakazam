# Python bindings for boa — compiled with pybind11.
# Produces _boa.so (or _boa.cpython-3xx-...so) importable from Python.
#
# NO_EXTRAS disables pybind11's LTO, which conflicts with CUDA's fatbinData
# symbols when linking against libboa (a CUDA library).

pybind11_add_module(_boa NO_EXTRAS pybind_module.cpp)
target_link_libraries(_boa PRIVATE boa)
target_compile_features(_boa PRIVATE cxx_std_17)
