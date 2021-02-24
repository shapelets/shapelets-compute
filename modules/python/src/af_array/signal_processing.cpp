#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void signal_processing_bindings(py::module &m) {

    m.def("convolve",
          [](const py::object &signal, const py::object &filter, const af::convMode mode, const af::convDomain domain) {
              af::array s;
              af::array f;
              if (py::isinstance<af::array>(signal))
                  s = py::cast<af::array>(signal);
              else
                  s = af_from_array_like(signal, std::nullopt, std::nullopt);

              if (py::isinstance<af::array>(filter))
                  f = py::cast<af::array>(filter);
              else
                  f = af_from_array_like(filter, std::nullopt, std::nullopt);

              return af::convolve(s, f, mode, domain);
          },
          py::arg("signal").none(false),
          py::arg("filter").none(false),
          py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
          py::arg("domain") = af::convDomain::AF_CONV_AUTO,
          "TODO");

//
//    m.def("convolve",
//          [](const af::array &signal, const af::array &filter, const af::convMode mode, const af::convDomain domain) {
//              return af::convolve(signal, filter, mode, domain);
//          },
//          py::arg("signal").none(false),
//          py::arg("filter").none(false),
//          py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
//          py::arg("domain") = af::convDomain::AF_CONV_AUTO,
//          "TODO");

    m.def("convolve1",
          [](const af::array &signal, const af::array &filter, const af::convMode mode, const af::convDomain domain) {
              return af::convolve1(signal, filter, mode, domain);
          },
          py::arg("signal").none(false),
          py::arg("filter").none(false),
          py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
          py::arg("domain") = af::convDomain::AF_CONV_AUTO,
          "TODO");

    m.def("convolve2",
          [](const af::array &signal, const af::array &filter, const af::convMode mode, const af::convDomain domain) {
              return af::convolve2(signal, filter, mode, domain);
          },
          py::arg("signal").none(false),
          py::arg("filter").none(false),
          py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
          py::arg("domain") = af::convDomain::AF_CONV_AUTO,
          "TODO");

    m.def("convolve3",
          [](const af::array &signal, const af::array &filter, const af::convMode mode, const af::convDomain domain) {
              return af::convolve3(signal, filter, mode, domain);
          },
          py::arg("signal").none(false),
          py::arg("filter").none(false),
          py::arg("mode") = af::convMode::AF_CONV_DEFAULT,
          py::arg("domain") = af::convDomain::AF_CONV_AUTO,
          "TODO");

    m.def("fft",
          [](const af::array &signal, const std::optional<py::int_> &odim, const std::optional<py::float_> &norm) {
              if (norm.has_value())
                  return af::fftNorm(signal, norm.value(), odim.value_or(0));

              return af::fft(signal, odim.value_or(0));
          },
          py::arg("signal").none(false),
          py::arg("odim") = 0,
          py::arg("norm") = py::none(),
          R"__(Fast fourier transform on one dimensional signals.
            Parameters
            ----------
                signal: One dimensional array.  Required
                odim  : Integer, Defaults to zero.
                        The length of the output signal, in order to truncate or pad the input signal
                norm  : Float, Defaults to none.
                        The scaling factor; if not provided, it is computed internally.
          )__");

    m.def("ifft",
          [](const af::array &coeff, const std::optional<py::int_> &odim, const std::optional<py::float_> &norm) {
              if (norm.has_value())
                  return af::ifftNorm(coeff, norm.value(), odim.value_or(0));

              return af::ifft(coeff, odim.value_or(0));
          },
          py::arg("signal").none(false),
          py::arg("odim") = 0,
          py::arg("norm") = py::none(),
          R"__(Fast fourier transform on one dimensional signals.
            Parameters
            ----------
                coeff : FFT coefficients.
                odim  : Integer, Defaults to zero.
                        The length of the output signal, in order to truncate or pad the input signal
                norm  : Float, Defaults to none.
                        The scaling factor; if not provided, it is computed internally.
          )__");

}