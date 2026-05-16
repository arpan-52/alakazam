// pybind_module.cpp — Python bindings for boa calibration solvers.
//
// Exposes solve_G, solve_K, solve_D, solve_KC, solve_CP as Python functions.
// Inputs are numpy arrays; outputs are Python dicts with numpy arrays.
//
// numpy complex128 ↔ Kokkos::complex<double> share binary layout (real, imag
// as adjacent doubles), so element-wise copy through host mirrors is safe.
//
// Kokkos is initialized once at module import and finalized via Python atexit.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boa/types.hpp>
#include <boa/api.hpp>
#include <Kokkos_Core.hpp>
#include <cstring>

namespace py = pybind11;
using namespace boa;

// ---------------------------------------------------------------------------
// numpy → Kokkos device view
// ---------------------------------------------------------------------------

static view_2d_complex np_to_view_2d_complex(
    py::array_t<std::complex<double>, py::array::c_style> arr, const char* name)
{
    auto buf = arr.request();
    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);
    view_2d_complex dev(name, rows, cols);
    auto hst = Kokkos::create_mirror_view(dev);
    const auto* src = static_cast<const complex_type*>(buf.ptr);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            hst(i, j) = src[i * cols + j];
    Kokkos::deep_copy(dev, hst);
    return dev;
}

static view_1d_int np_to_view_1d_int(
    py::array_t<int, py::array::c_style> arr, const char* name)
{
    auto buf = arr.request();
    const int n = static_cast<int>(buf.shape[0]);
    view_1d_int dev(name, n);
    auto hst = Kokkos::create_mirror_view(dev);
    const int* src = static_cast<const int*>(buf.ptr);
    for (int i = 0; i < n; ++i) hst(i) = src[i];
    Kokkos::deep_copy(dev, hst);
    return dev;
}

static view_1d_real np_to_view_1d_real(
    py::array_t<double, py::array::c_style> arr, const char* name)
{
    auto buf = arr.request();
    const int n = static_cast<int>(buf.shape[0]);
    view_1d_real dev(name, n);
    auto hst = Kokkos::create_mirror_view(dev);
    const double* src = static_cast<const double*>(buf.ptr);
    for (int i = 0; i < n; ++i) hst(i) = src[i];
    Kokkos::deep_copy(dev, hst);
    return dev;
}

// ---------------------------------------------------------------------------
// Kokkos device view → numpy
// ---------------------------------------------------------------------------

static py::array_t<std::complex<double>> view_2d_complex_to_np(
    const view_2d_complex& dev)
{
    const int rows = static_cast<int>(dev.extent(0));
    const int cols = static_cast<int>(dev.extent(1));
    py::array_t<std::complex<double>> out({rows, cols});
    auto buf = out.request();
    auto hst = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dev);
    auto* dst = static_cast<complex_type*>(buf.ptr);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i * cols + j] = hst(i, j);
    return out;
}

static py::array_t<double> view_1d_real_to_np(const view_1d_real& dev)
{
    const int n = static_cast<int>(dev.extent(0));
    py::array_t<double> out(n);
    auto buf = out.request();
    auto hst = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dev);
    auto* dst = static_cast<double*>(buf.ptr);
    for (int i = 0; i < n; ++i) dst[i] = hst(i);
    return out;
}

// ---------------------------------------------------------------------------
// Build SolverInput from Python arguments
// ---------------------------------------------------------------------------

static SolverInput make_input(
    py::array_t<std::complex<double>, py::array::c_style> vis_obs_np,
    py::array_t<std::complex<double>, py::array::c_style> vis_model_np,
    py::array_t<int, py::array::c_style> ant1_np,
    py::array_t<int, py::array::c_style> ant2_np,
    py::array_t<double, py::array::c_style> freqs_np,
    int n_ant, int ref_ant)
{
    SolverInput inp;
    inp.vis_obs   = np_to_view_2d_complex(vis_obs_np,   "vis_obs");
    inp.vis_model = np_to_view_2d_complex(vis_model_np, "vis_model");
    inp.ant1      = np_to_view_1d_int(ant1_np, "ant1");
    inp.ant2      = np_to_view_1d_int(ant2_np, "ant2");
    inp.freqs     = np_to_view_1d_real(freqs_np, "freqs");
    inp.n_ant     = n_ant;
    inp.ref_ant   = ref_ant;
    return inp;
}

// ---------------------------------------------------------------------------
// Pack SolverResult into a Python dict
// ---------------------------------------------------------------------------

static py::dict pack_result(const SolverResult& res)
{
    py::dict d;
    d["jones"]     = view_2d_complex_to_np(res.jones);
    d["params"]    = view_1d_real_to_np(res.params);
    d["cost"]      = res.cost;
    d["n_iter"]    = res.n_iter;
    d["converged"] = res.converged;
    return d;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_boa, m)
{
    m.doc() = "boa — Kokkos calibration solver Python bindings";

    // Initialize Kokkos once; finalize via Python atexit.
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        py::module_::import("atexit").attr("register")(
            py::cpp_function([]() {
                if (Kokkos::is_initialized()) Kokkos::finalize();
            }));
    }

    // Report the Kokkos default execution space
    m.def("execution_space", []() -> std::string {
        return Kokkos::DefaultExecutionSpace::name();
    }, "Return the Kokkos default execution space (e.g., 'Cuda', 'OpenMP', 'Serial').");

    // SolverOptions. linear_solver is exposed as a string:
    //   "cholesky" → DENSE_CHOLESKY, "cg" → CG, "lsqr" (default) → LSQR.
    py::class_<SolverOptions>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("max_iter",        &SolverOptions::max_iter)
        .def_readwrite("tol",             &SolverOptions::tol)
        .def_readwrite("linear_max_iter", &SolverOptions::linear_max_iter)
        .def_readwrite("linear_tol",      &SolverOptions::linear_tol)
        .def_readwrite("lm_lambda_init",  &SolverOptions::lm_lambda_init)
        .def_property("linear_solver",
            [](const SolverOptions& s) -> std::string {
                switch (s.linear_solver) {
                    case LinearSolverType::DENSE_CHOLESKY: return "cholesky";
                    case LinearSolverType::CG:             return "cg";
                    default:                               return "lsqr";
                }
            },
            [](SolverOptions& s, const std::string& v) {
                if (v == "cholesky")
                    s.linear_solver = LinearSolverType::DENSE_CHOLESKY;
                else if (v == "cg")
                    s.linear_solver = LinearSolverType::CG;
                else
                    s.linear_solver = LinearSolverType::LSQR;
            });

    // Common docstring for all solve_* functions.
    // Args:
    //   vis_obs   : (n_vis, 4) complex128 — observed visibilities (pp, pq, qp, qq)
    //   vis_model : (n_vis, 4) complex128 — model visibilities
    //   ant1      : (n_bl,) int32 — first antenna index per baseline
    //   ant2      : (n_bl,) int32 — second antenna index per baseline
    //   freqs     : (n_freq,) float64 — channel frequencies in Hz (empty for freq-indep)
    //   n_ant     : int — total number of antennas
    //   ref_ant   : int — reference antenna (fixed at identity)
    //   opts      : SolverOptions (optional)
    // Returns dict with keys: jones, params, cost, n_iter, converged.

    m.def("solve_G",
        [](py::array_t<std::complex<double>, py::array::c_style> vis_obs,
           py::array_t<std::complex<double>, py::array::c_style> vis_model,
           py::array_t<int, py::array::c_style> ant1,
           py::array_t<int, py::array::c_style> ant2,
           py::array_t<double, py::array::c_style> freqs,
           int n_ant, int ref_ant, const SolverOptions& opts,
           py::array_t<double, py::array::c_style> init_params) {
            auto inp = make_input(vis_obs, vis_model, ant1, ant2, freqs, n_ant, ref_ant);
            if (init_params.size() > 0)
                inp.init_params = np_to_view_1d_real(init_params, "g_init");
            return pack_result(solve_G(inp, opts));
        },
        py::arg("vis_obs"), py::arg("vis_model"),
        py::arg("ant1"), py::arg("ant2"), py::arg("freqs"),
        py::arg("n_ant"), py::arg("ref_ant"),
        py::arg("opts") = SolverOptions(),
        py::arg("init_params") = py::array_t<double>(),
        "Solve for diagonal gain Jones matrices (G solver).\n"
        "Returns dict: jones (n_ant,4), params, cost, n_iter, converged.");

    m.def("solve_K",
        [](py::array_t<std::complex<double>, py::array::c_style> vis_obs,
           py::array_t<std::complex<double>, py::array::c_style> vis_model,
           py::array_t<int, py::array::c_style> ant1,
           py::array_t<int, py::array::c_style> ant2,
           py::array_t<double, py::array::c_style> freqs,
           int n_ant, int ref_ant, const SolverOptions& opts,
           py::array_t<double, py::array::c_style> init_params) {
            auto inp = make_input(vis_obs, vis_model, ant1, ant2, freqs, n_ant, ref_ant);
            if (init_params.size() > 0)
                inp.init_params = np_to_view_1d_real(init_params, "k_init");
            return pack_result(solve_K(inp, opts));
        },
        py::arg("vis_obs"), py::arg("vis_model"),
        py::arg("ant1"), py::arg("ant2"), py::arg("freqs"),
        py::arg("n_ant"), py::arg("ref_ant"),
        py::arg("opts") = SolverOptions(),
        py::arg("init_params") = py::array_t<double>(),
        "Solve for per-antenna delays (K solver).\n"
        "Returns dict: params = [tau_p, tau_q] per non-ref antenna in ns, cost, n_iter, converged.");

    m.def("solve_D",
        [](py::array_t<std::complex<double>, py::array::c_style> vis_obs,
           py::array_t<std::complex<double>, py::array::c_style> vis_model,
           py::array_t<int, py::array::c_style> ant1,
           py::array_t<int, py::array::c_style> ant2,
           py::array_t<double, py::array::c_style> freqs,
           int n_ant, int ref_ant, const SolverOptions& opts,
           py::array_t<double, py::array::c_style> init_params) {
            auto inp = make_input(vis_obs, vis_model, ant1, ant2, freqs, n_ant, ref_ant);
            if (init_params.size() > 0)
                inp.init_params = np_to_view_1d_real(init_params, "d_init");
            return pack_result(solve_D(inp, opts));
        },
        py::arg("vis_obs"), py::arg("vis_model"),
        py::arg("ant1"), py::arg("ant2"), py::arg("freqs"),
        py::arg("n_ant"), py::arg("ref_ant"),
        py::arg("opts") = SolverOptions(),
        py::arg("init_params") = py::array_t<double>(),
        "Solve for leakage Jones matrices (D solver).\n"
        "Returns dict: jones (n_ant,4) full 2x2, params, cost, n_iter, converged.");

    m.def("solve_KC",
        [](py::array_t<std::complex<double>, py::array::c_style> vis_obs,
           py::array_t<std::complex<double>, py::array::c_style> vis_model,
           py::array_t<int, py::array::c_style> ant1,
           py::array_t<int, py::array::c_style> ant2,
           py::array_t<double, py::array::c_style> freqs,
           int n_ant, int ref_ant, const SolverOptions& opts,
           py::array_t<double, py::array::c_style> init_params) {
            auto inp = make_input(vis_obs, vis_model, ant1, ant2, freqs, n_ant, ref_ant);
            if (init_params.size() > 0)
                inp.init_params = np_to_view_1d_real(init_params, "kc_init");
            return pack_result(solve_KC(inp, opts));
        },
        py::arg("vis_obs"), py::arg("vis_model"),
        py::arg("ant1"), py::arg("ant2"), py::arg("freqs"),
        py::arg("n_ant"), py::arg("ref_ant"),
        py::arg("opts") = SolverOptions(),
        py::arg("init_params") = py::array_t<double>(),
        "Solve for global cross-delay tau (KC solver).\n"
        "Returns dict: params = [tau_ns], cost, n_iter, converged.");

    m.def("solve_CP",
        [](py::array_t<std::complex<double>, py::array::c_style> vis_obs,
           py::array_t<std::complex<double>, py::array::c_style> vis_model,
           py::array_t<int, py::array::c_style> ant1,
           py::array_t<int, py::array::c_style> ant2,
           py::array_t<double, py::array::c_style> freqs,
           int n_ant, int ref_ant, const SolverOptions& opts,
           py::array_t<double, py::array::c_style> init_params) {
            auto inp = make_input(vis_obs, vis_model, ant1, ant2, freqs, n_ant, ref_ant);
            if (init_params.size() > 0)
                inp.init_params = np_to_view_1d_real(init_params, "cp_init");
            return pack_result(solve_CP(inp, opts));
        },
        py::arg("vis_obs"), py::arg("vis_model"),
        py::arg("ant1"), py::arg("ant2"), py::arg("freqs"),
        py::arg("n_ant"), py::arg("ref_ant"),
        py::arg("opts") = SolverOptions(),
        py::arg("init_params") = py::array_t<double>(),
        "Solve for global cross-phase phi (CP solver).\n"
        "Returns dict: params = [phi_rad], cost, n_iter, converged.");
}
