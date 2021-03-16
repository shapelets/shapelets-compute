#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

namespace py = pybind11;

void pygauss::bindings::signal_processing_functions(py::module &m)
{

    m.def(
        "convolve",
        [](const py::object &signal, const py::object &filter, const af::convMode mode, const af::convDomain domain) {
            af::array s = arraylike::as_array_checked(signal);
            af::array f = arraylike::as_array_checked(filter);

            arraylike::ensure_floating(s);
            arraylike::ensure_floating(f);

            return af::convolve(s, f, mode, domain);
        },
        py::arg("signal").none(false),
        py::arg("filter").none(false),
        py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
        py::arg("domain") = af::convDomain::AF_CONV_AUTO,
        "TODO");

    m.def(
        "convolve1",
        [](const py::object &signal, const py::object &filter, const af::convMode mode, const af::convDomain domain) {
            af::array s = arraylike::as_array_checked(signal);
            af::array f = arraylike::as_array_checked(filter);
            arraylike::ensure_floating(s);
            arraylike::ensure_floating(f);
            return af::convolve1(s, f, mode, domain);
        },
        py::arg("signal").none(false),
        py::arg("filter").none(false),
        py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
        py::arg("domain") = af::convDomain::AF_CONV_AUTO,
        "TODO");

    m.def(
        "convolve2",
        [](const py::object &signal, const py::object &filter, const af::convMode mode, const af::convDomain domain) {
            af::array s = arraylike::as_array_checked(signal);
            af::array f = arraylike::as_array_checked(filter);
            arraylike::ensure_floating(s);
            arraylike::ensure_floating(f);
            return af::convolve2(s, f, mode, domain);
        },
        py::arg("signal").none(false),
        py::arg("filter").none(false),
        py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
        py::arg("domain") = af::convDomain::AF_CONV_AUTO,
        "TODO");

    m.def(
        "convolve3",
        [](const py::object &signal, const py::object &filter, const af::convMode mode, const af::convDomain domain) {
            af::array s = arraylike::as_array_checked(signal);
            af::array f = arraylike::as_array_checked(filter);
            arraylike::ensure_floating(s);
            arraylike::ensure_floating(f);

            return af::convolve3(s, f, mode, domain);
        },
        py::arg("signal").none(false),
        py::arg("filter").none(false),
        py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
        py::arg("domain") = af::convDomain::AF_CONV_AUTO,
        "TODO");

    py::enum_<gauss::fft::Norm>(m, "fftNorm", "Gauss FFT Normalisation")
            .value("Backward", gauss::fft::Norm::Backward, "signal -> freq: 1.0, freq -> signal: 1.0/n")
            .value("Ortho", gauss::fft::Norm::Orthonormal, "signal -> freq: 1.0/sqrt(n), freq -> signal: 1.0/sqrt(n)")
            .value("Forward", gauss::fft::Norm::Forward, "signal -> freq: 1.0/n, freq -> signal: 1.0")
            .export_values();

    m.def(
        "_fft",
        [](const py::object &signal, const std::variant<gauss::fft::Norm, double>& norm, const af::dim4& shape) {
            af::array s = arraylike::as_array_checked(signal);
            arraylike::ensure_floating(s);

            return gauss::fft::fft(s, norm, shape);
        },
        py::arg("signal").none(false),
        py::arg("norm").none(false),
        py::arg("shape").none(false),
        "TODO");

    m.def(
        "_ifft",
        [](const py::object &coef, const std::variant<gauss::fft::Norm, double>& norm, const af::dim4& shape) {
            af::array c = arraylike::as_array_checked(coef);
            arraylike::ensure_floating(c);
            return gauss::fft::ifft(c, norm, shape);
        },
        py::arg("coef").none(false),
        py::arg("norm").none(false),
        py::arg("shape").none(false),
        "TODO");

    m.def(
        "_rfft",
        [](const py::object &signal, const std::variant<gauss::fft::Norm, double>& norm, const af::dim4& shape) {
            af::array s = arraylike::as_array_checked(signal);
            arraylike::ensure_floating(s);
            return gauss::fft::rfft(s, norm, shape);
        },
        py::arg("signal").none(false),
        py::arg("norm").none(false),
        py::arg("shape").none(false),
        "TODO");

    m.def(
        "_irfft",
        [](const py::object &coef, const std::variant<gauss::fft::Norm, double>& norm, const af::dim4& shape) {
            af::array c = arraylike::as_array_checked(coef);
            arraylike::ensure_floating(c);
            return gauss::fft::irfft(c, norm, shape);
        },
        py::arg("coef").none(false),
        py::arg("norm").none(false),
        py::arg("shape").none(false),
        "TODO");

    m.def(
        "rfftfreq",
        [](const int n, const double d) {
            return gauss::fft::rfftfreq(n, d);
        },
        py::arg("n").none(false),
        py::arg("d") = 1.0,
        "");

    m.def("fftfreq",         
        [](const int n, const double d) {
            return gauss::fft::fftfreq(n, d);
        },
        py::arg("n").none(false),
        py::arg("d") = 1.0,
        "");
}

