/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

#include <optional>

namespace py = pybind11;

void pygauss::bindings::signal_processing_functions(py::module &m)
{
    py::enum_<gauss::fft::Norm>(m, "fftNorm", "Gauss FFT Normalisation")
            .value("Backward", gauss::fft::Norm::Backward, "signal -> freq: 1.0, freq -> signal: 1.0/n")
            .value("Ortho", gauss::fft::Norm::Orthonormal, "signal -> freq: 1.0/sqrt(n), freq -> signal: 1.0/sqrt(n)")
            .value("Forward", gauss::fft::Norm::Forward, "signal -> freq: 1.0/n, freq -> signal: 1.0")
            .export_values();

    m.def(
        "fft",
        [](const py::object &signal, const std::variant<gauss::fft::Norm, double>& norm, const std::optional<af::dim4>& shape) {
            af::array s = arraylike::as_array_checked(signal);
            arraylike::ensure_floating(s);
            auto checked_shape = shape.value_or(s.dims());
            return gauss::fft::fft(s, norm, checked_shape);
        },
        py::arg("signal").none(false),
        py::arg("norm").none(false),
        py::arg("shape") = std::nullopt);

    m.def(
        "ifft",
        [](const py::object &coef, const std::variant<gauss::fft::Norm, double>& norm, const std::optional<af::dim4>& shape) {
            af::array c = arraylike::as_array_checked(coef);
            arraylike::ensure_floating(c);
            auto checked_shape = shape.value_or(c.dims());
            return gauss::fft::ifft(c, norm, checked_shape);
        },
        py::arg("coef").none(false),
        py::arg("norm").none(false),
        py::arg("shape") = std::nullopt);

    m.def(
        "rfft",
        [](const py::object &signal, const std::variant<gauss::fft::Norm, double>& norm, const std::optional<af::dim4>& shape) {
            af::array s = arraylike::as_array_checked(signal);
            arraylike::ensure_floating(s);
            auto checked_shape = shape.value_or(s.dims());
            return gauss::fft::rfft(s, norm, checked_shape);
        },
        py::arg("signal").none(false),
        py::arg("norm").none(false),
        py::arg("shape") = std::nullopt);

    m.def(
        "irfft",
        [](const py::object &coef, const std::variant<gauss::fft::Norm, double>& norm, const std::optional<af::dim4>& shape) {
            af::array c = arraylike::as_array_checked(coef);
            arraylike::ensure_floating(c);
            auto checked_shape = shape.value_or(c.dims());
            return gauss::fft::irfft(c, norm, checked_shape);
        },
        py::arg("coef").none(false),
        py::arg("norm").none(false),
        py::arg("shape") = std::nullopt);

    m.def(
        "rfftfreq",
        [](const int n, const double d) {
            return gauss::fft::rfftfreq(n, d);
        },
        py::arg("n").none(false),
        py::arg("d") = 1.0);

    m.def("fftfreq",         
        [](const int n, const double d) {
            return gauss::fft::fftfreq(n, d);
        },
        py::arg("n").none(false),
        py::arg("d") = 1.0);

    m.def("spectral_derivative",
        [](const py::object &signal, const py::object& kappa_spec, const bool shift) {
            af::array s = arraylike::as_array_checked(signal);
            arraylike::ensure_floating(s);
            std::variant<double, af::array> ks;
            if (py::isinstance<py::float_>(kappa_spec)) {
                ks = kappa_spec.cast<double>();
            }
            else if(py::isinstance<py::int_>(kappa_spec)) {
                ks = kappa_spec.cast<double>();
            } 
            else {
                auto ks_array = arraylike::as_array_checked(kappa_spec);
                arraylike::ensure_floating(ks_array);
                ks = ks_array;
            }
            return gauss::fft::spectral_derivative(s, ks, shift);
        },
        py::arg("signal").none(false),
        py::arg("kappa_spec") = 1.0,
        py::arg("shift") = true);

    m.def("fftshift",
        [](const py::object &x, const std::optional<std::variant<int, std::vector<int>>>& axes) {
            af::array s = arraylike::as_array_checked(x);
            arraylike::ensure_floating(s);
            return gauss::fft::fftshift(s, axes);
        },
        py::arg("x").none(false),
        py::arg("axes") = py::none());
}

