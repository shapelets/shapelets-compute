#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void statistic_bindings(py::module &m) {

    m.def("covs",
          [](const af::array &a, const af::array &b) {
              return af::cov(a, b, af_var_bias::AF_VARIANCE_SAMPLE);
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          "Find the covariance (sample) of values between two arrays.");

    m.def("covp",
          [](const af::array &a, const af::array &b) {
              return af::cov(a, b, af_var_bias::AF_VARIANCE_SAMPLE);
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          "Find the covariance (population) of values between two arrays.");

    m.def("corrcoef",
          [](const af::array &a, const af::array &b) {
              std::variant<double, std::complex<double>> result;
              if (a.iscomplex() || b.iscomplex()) {
                  double real;
                  double imag;
                  check_af_error(af_corrcoef(&real, &imag, a.get(), b.get()));
                  result = std::complex<double>(real, imag);
              } else {
                  result = af::corrcoef<double>(a, b);
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          "Computes Pearson product-moment correlation coefficient.");

    m.def("topk_max",
          [](const af::array &a, const int k) {
              af::array values;
              af::array indices;
              af::topk(values, indices, k, 0, af_topk_function::AF_TOPK_MAX);
              py::tuple result(2);
              result[0] = values;
              result[1] = indices;
              return result;
          },
          py::arg("a").none(false),
          py::arg("k").none(false),
          "The top k max values along a given dimension of the input array");


    m.def("topk_min",
          [](const af::array &a, const int k) {
              af::array values;
              af::array indices;
              af::topk(values, indices, k, 0, af_topk_function::AF_TOPK_MIN);

              py::tuple result(2);
              result[0] = values;
              result[1] = indices;
              return result;
          },
          py::arg("a").none(false),
          py::arg("k").none(false),
          "The top k max values along a given dimension of the input array");

    m.def("mean",
          [](const af::array &a, const std::optional<af::array> &weights, const std::optional<int> dim) {

              auto hasWeights = weights.has_value();
              auto isAggregated = !dim.has_value();
              std::variant<double, std::complex<double>, af::array> result;

              if (isAggregated) {
                  double real, imag;
                  af_err err;
                  if (hasWeights)
                      err = af_mean_all_weighted(&real, &imag, a.get(), weights->get());
                  else
                      err = af_mean_all(&real, &imag, a.get());

                  check_af_error(err);

                  if (a.iscomplex())
                      result = std::complex<double>(real, imag);
                  else
                      result = real;
              } else {
                  if (hasWeights)
                      result = af::mean(a, weights.value(), dim.value());
                  else
                      result = af::mean(a, dim.value());
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("weights") = py::none(),
          py::arg("dim") = py::none(),
          "Computes mean on an array."
          "\n"
          "When the parameter dim is unset, it computes the mean across all values in the matrix.  When dim "
          "has a value, it computes the mean across a particular dimension; if dim is -1, the mean will be "
          "produced over the first non trivial dimension."
          "\n"
          "The result of this computation is either an array, or a scalar value (complex or float)");

    m.def("median",
          [](const af::array &a, const std::optional<int> dim) {

              auto isAggregated = !dim.has_value();
              std::variant<double, std::complex<double>, af::array> result;

              if (isAggregated) {
                  double real, imag;
                  check_af_error(af_median_all(&real, &imag, a.get()));

                  if (a.iscomplex())
                      result = std::complex<double>(real, imag);
                  else
                      result = real;
              } else {
                  result = af::mean(a, dim.value());
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("dim") = py::none(),
          "Computes median on an array."
          "\n"
          "When the parameter dim is unset, it computes the median across all values in the matrix.  When dim "
          "has a value, it computes the median across a particular dimension; if dim is -1, the median will be "
          "produced over the first non trivial dimension."
          "\n"
          "The result of this computation is either an array, or a scalar value (complex or float)");

    m.def("stdev",
          [](const af::array &a, const std::optional<int> dim) {
              std::variant<double, std::complex<double>, af::array> result;
              auto isAggregated = !dim.has_value();

              if (isAggregated) {
                  double real, imag;
                  check_af_error(af_stdev_all(&real, &imag, a.get()));

                  if (a.iscomplex())
                      result = std::complex<double>(real, imag);
                  else
                      result = real;
              } else {
                  result = af::stdev(a, dim.value());
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("dim") = py::none(),
          "Computes stdev on an array.");

    m.def("var_p",
          [](const af::array &a, const std::optional<af::array> &weights, const std::optional<int> dim) {

              auto hasWeights = weights.has_value();
              auto isAggregated = !dim.has_value();
              std::variant<double, std::complex<double>, af::array> result;

              if (isAggregated) {
                  double real, imag;
                  af_err err;
                  if (hasWeights)
                      err = af_var_all_weighted(&real, &imag, a.get(), weights->get());
                  else
                      err = af_var_all(&real, &imag, a.get(), false);

                  check_af_error(err);

                  if (a.iscomplex())
                      result = std::complex<double>(real, imag);
                  else
                      result = real;
              } else {
                  if (hasWeights)
                      result = af::var(a, weights.value(), dim.value());
                  else
                      result = af::var(a, false, dim.value());
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("weights") = py::none(),
          py::arg("dim") = py::none());

    m.def("var_s",
          [](const af::array &a, const std::optional<af::array> &weights, const std::optional<int> dim) {
              auto hasWeights = weights.has_value();
              auto isAggregated = !dim.has_value();
              std::variant<double, std::complex<double>, af::array> result;

              if (isAggregated) {
                  double real, imag;
                  af_err err;
                  if (hasWeights)
                      err = af_var_all_weighted(&real, &imag, a.get(), weights->get());
                  else
                      err = af_var_all(&real, &imag, a.get(), true);

                  check_af_error(err);

                  if (a.iscomplex())
                      result = std::complex<double>(real, imag);
                  else
                      result = real;
              } else {
                  if (hasWeights)
                      result = af::var(a, weights.value(), dim.value());
                  else
                      result = af::var(a, true, dim.value());
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("weights") = py::none(),
          py::arg("dim") = py::none());
}