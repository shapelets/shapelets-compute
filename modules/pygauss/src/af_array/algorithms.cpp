#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

typedef af_err (*reduce_all_op)(double *, double *, af_array);

typedef af_err (*reduce_all_nan_op)(double *, double *, af_array, double);

typedef af_err (*reduce_dim_op)(af_array *, af_array, const int);

typedef af_err (*reduce_dim_nan_op)(af_array *, af_array, const int, const double);

std::complex<double> reduce_all_complex(const af::array &a, const reduce_all_op op) {
    double real, imag;
    check_af_error((*op)(&real, &imag, a.get()));
    return std::complex<double>(real, imag);
}

std::complex<double> reduce_all_complex_nan(const af::array &a, const double nan, const reduce_all_nan_op op) {
    double real, imag;
    check_af_error((*op)(&real, &imag, a.get(), nan));
    return std::complex<double>(real, imag);
}


double reduce_all_real(const af::array &a, const reduce_all_op op) {
    double real, imag;
    check_af_error((*op)(&real, &imag, a.get()));
    return real;
}

double reduce_all_real_nan(const af::array &a, const double nan, const reduce_all_nan_op op) {
    double real, imag;
    check_af_error((*op)(&real, &imag, a.get(), nan));
    return real;
}

af::array reduce_dim(const af::array &a, const int dim, const reduce_dim_op op) {
    af_array result = nullptr;
    check_af_error((*op)(&result, a.get(), dim));
    return af::array(result);
}

af::array reduce_dim_nan(const af::array &a, const int dim, double nan, const reduce_dim_nan_op op) {
    af_array result = nullptr;
    check_af_error((*op)(&result, a.get(), dim, nan));
    return af::array(result);
}

void algorithm_bindings(py::module &m) {

    // TODO: missing by key* algos

    m.def("min",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<double, std::complex<double>, af::array> result;
              if (!dim.has_value()) {
                  if (a.iscomplex())
                      result = reduce_all_complex(a, af_min_all);
                  else
                      result = reduce_all_real(a, af_min_all);
              } else {
                  result = reduce_dim(a, dim.value(), af_min);
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          "Find the minimum value of all the elements along a specified dimension.");

    // todo: missing imax and imin

    m.def("argmax",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<std::tuple<unsigned int, double>,
                      std::tuple<unsigned int, std::complex<double>>,
                      std::tuple<af::array, af::array>> result;

              if (!dim.has_value()) {
                  double real, imag;
                  unsigned int index;
                  check_af_error(af_imax_all(&real, &imag, &index, a.get()));
                  if (a.iscomplex())
                      result = std::make_tuple(index, std::complex<double>(real, imag));
                  else
                      result = std::make_tuple(index, real);
              }
              else {
                  af_array out = nullptr;
                  af_array index = nullptr;
                  check_af_error(af_imax(&out, &index, a.get(), dim.value()));
                  result = std::make_tuple(af::array(index), af::array(out));
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim") = py::none(),
          "TODO");

    m.def("argmin",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<std::tuple<unsigned int, double>,
                      std::tuple<unsigned int, std::complex<double>>,
                      std::tuple<af::array, af::array>> result;

              if (!dim.has_value()) {
                  double real, imag;
                  unsigned int index;
                  check_af_error(af_imin_all(&real, &imag, &index, a.get()));
                  if (a.iscomplex())
                      result = std::make_tuple(index, std::complex<double>(real, imag));
                  else
                      result = std::make_tuple(index, real);
              }
              else {
                  af_array out = nullptr;
                  af_array index = nullptr;
                  check_af_error(af_imin(&out, &index, a.get(), dim.value()));
                  result = std::make_tuple(af::array(index), af::array(out));
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim") = py::none(),
          "TODO");

    m.def("max",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<double, std::complex<double>, af::array> result;
              if (!dim.has_value()) {
                  if (a.iscomplex())
                      result = reduce_all_complex(a, af_max_all);
                  else
                      result = reduce_all_real(a, af_max_all);
              } else {
                  result = reduce_dim(a, dim.value(), af_max);
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          "Find the maximum value of all the elements along a specified dimension.");

    m.def("count",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<double, af::array> result;
              if (!dim.has_value())
                  result = reduce_all_real(a, af_count_all);
              else
                  result = reduce_dim(a, dim.value(), af_count);
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          "Count the number of non zero elements in an array along a specified dimension");

    m.def("any_true",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<bool, af::array> result;
              if (!dim.has_value())
                  result = reduce_all_real(a, af_any_true_all) == 1.0;
              else {
                  result = reduce_dim(a, dim.value(), af_any_true);
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          "Check if any the elements along a specified dimension are true.");

    m.def("all_true",
          [](const af::array &a, const std::optional<int> &dim) {
              std::variant<bool, af::array> result;
              if (!dim.has_value())
                  result = reduce_all_real(a, af_all_true_all) == 1.0;
              else {
                  result = reduce_dim(a, dim.value(), af_all_true);
              }
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          "Check if all the elements along a specified dimension are true.");

    m.def("sum",
          [](const af::array &a, const std::optional<int> &dim, const std::optional<double> &nan_value) {
              std::variant<double, std::complex<double>, af::array> result;

              if (!dim.has_value()) {
                  if (!nan_value.has_value())
                      if (a.iscomplex())
                          result = reduce_all_complex(a, af_sum_all);
                      else
                          result = reduce_all_real(a, af_sum_all);
                  else if (a.iscomplex())
                      result = reduce_all_complex_nan(a, nan_value.value(), af_sum_nan_all);
                  else
                      result = reduce_all_real_nan(a, nan_value.value(), af_sum_nan_all);
              } else {
                  if (!nan_value.has_value()) {
                      result = reduce_dim(a, dim.value(), af_sum);
                  } else {
                      result = reduce_dim_nan(a, dim.value(), nan_value.value(), af_sum_nan);
                  }
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          py::arg("nan_value").none(true) = py::none(),
          "Calculate the sum of all the elements along a specified dimension.");

    m.def("product",
          [](const af::array &a, const std::optional<int> &dim, const std::optional<double> &nan_value) {
              std::variant<double, std::complex<double>, af::array> result;

              if (!dim.has_value()) {
                  if (!nan_value.has_value())
                      if (a.iscomplex())
                          result = reduce_all_complex(a, af_product_all);
                      else
                          result = reduce_all_real(a, af_product_all);
                  else if (a.iscomplex())
                      result = reduce_all_complex_nan(a, nan_value.value(), af_product_nan_all);
                  else
                      result = reduce_all_real_nan(a, nan_value.value(), af_product_nan_all);
              } else {
                  if (!nan_value.has_value()) {
                      result = reduce_dim(a, dim.value(), af_product);
                  } else {
                      result = reduce_dim_nan(a, dim.value(), nan_value.value(), af_product_nan);
                  }
              }

              return result;
          },
          py::arg("a").none(false),
          py::arg("dim").none(true) = py::none(),
          py::arg("nan_value").none(true) = py::none(),
          "Calculate the product of all the elements along a specified dimension.");

    m.def("cumsum",
          [](const af::array &a, const int &dim) {
              return reduce_dim(a, dim, af_accum);
          },
          py::arg("a").none(false),
          py::arg("dim") = 0,
          "Cumulative sum of an array along a specified dimension");

    m.def("scan",
          [](const af::array &a, const int &dim, const af::binaryOp &op, const bool &inclusive_scan) {
              return af::scan(a, dim, op, inclusive_scan);
          },
          py::arg("a").none(false),
          py::arg("dim") = 0,
          py::arg("op") = af::binaryOp::AF_BINARY_ADD,
          py::arg("inclusive_scan") = true,
          "Generalized scan of an array.");

    m.def("where",
          [](const af::array &a) { return af::where(a); },
          py::arg("a").none(false),
          "Find the indices of non zero elements");

    m.def("diff1",
          [](const af::array &a, const int &dim) {
              return reduce_dim(a, dim, af_diff1);
          },
          py::arg("a").none(false),
          py::arg("dim").none(false),
          "Find the first order differences along specified dimensions");

    m.def("diff2",
          [](const af::array &a, const int &dim) {
              return reduce_dim(a, dim, af_diff2);
          },
          py::arg("a").none(false),
          py::arg("dim").none(false),
          "Find the second order differences along specified dimensions");

    m.def("sort",
          [](const af::array &a, const int &dim, const bool &asc) {
              return af::sort(a, dim, asc);
          },
          py::arg("a").none(false),
          py::arg("dim") = 0,
          py::arg("asc") = py::bool_(true),
          "Sort the array along a specified dimension");

    m.def("sort_index",
          [](const af::array &a, const int &dim, const bool &asc) {
              af_array r, ri = nullptr;

              check_af_error(af_sort_index(&r, &ri, a.get(), dim, asc));

              py::tuple result(2);
              result[0] = af::array(r);
              result[1] = af::array(ri);
              return result;
          },
          py::arg("a").none(false),
          py::arg("dim") = 0,
          py::arg("asc") = py::bool_(true),
          "sorting an array and getting original indices");

    m.def("unique",
          [](const af::array &a, const bool &is_sorted) {
              return af::setUnique(a, is_sorted);
          }, py::arg("a").none(false),
          py::arg("is_sorted") = py::bool_(false),
          "Find the unique elements of an array.");

    m.def("union",
          [](const af::array &a, const af::array &b, const bool &is_unique) {
              return af::setUnion(a, b, is_unique);
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          py::arg("is_unique") = py::bool_(false),
          "Find the union of two arrays.");

    m.def("intersect",
          [](const af::array &a, const af::array &b, const bool &is_unique) {
              return af::setIntersect(a, b, is_unique);
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          py::arg("is_unique") = py::bool_(false),
          "Find the union of two arrays.");
}