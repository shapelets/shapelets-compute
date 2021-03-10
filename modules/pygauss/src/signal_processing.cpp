#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

namespace py = pybind11;

void pygauss::bindings::signal_processing_functions(py::module &m) {

    m.def("convolve",
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

    m.def("convolve1",
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

    m.def("convolve2",
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

    m.def("convolve3",
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

    m.def("fft",
          [](const py::object &signal, const std::optional<py::int_> &odim, const std::optional<py::float_> &norm) {
              af::array s = arraylike::as_array_checked(signal);
              arraylike::ensure_floating(s);

              if (norm.has_value())
                  return af::fftNorm(s, norm.value(), odim.value_or(0));

              return af::fft(s, odim.value_or(0));
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
          [](const py::object &coeff, const std::optional<py::int_> &odim, const std::optional<py::float_> &norm) {
              af::array c = arraylike::as_array_checked(coeff);
              arraylike::ensure_floating(c);

              if (norm.has_value())
                  return af::ifftNorm(c, norm.value(), odim.value_or(0));

              return af::ifft(c, odim.value_or(0));
          },
          py::arg("coeff").none(false),
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
