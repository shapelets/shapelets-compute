#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <af_array/af_array.h>

namespace py = pybind11;

void linear_algebra_bindings(py::module &m) {

    m.def("det",
          [](const af::array &a) {
              std::variant<py::float_, std::complex<double>> result;

              if (a.iscomplex()) {
                  double real, img = 0.0;
                  throw_on_error(af_det(&real, &img, a.get()));
                  result = std::complex<double>(real, img);
              } else {
                  auto checked = a;
                  if (!a.isfloating()) {
                      auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
                      PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
                      checked = a.as(safe_conversion);
                  }

                  if (a.type() == af::dtype::f64)
                      result = py::float_(af::det<double>(checked));
                  else
                      result = py::float_(af::det<float>(checked));
              }

              return result;
          }, py::arg("a").none(false),
          "Find the determinant of a matrix.\n"
          "If the matrix is an integer matrix, it will be internally casted to either a float or a double "
          "matrix, depending on the capabilities of the current device.");

    m.def("inverse",
          [](const af::array &a, const af_mat_prop options = af_mat_prop::AF_MAT_NONE) {
              auto src = a;
              if (!a.isfloating()) {
                  auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
                  PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
                  src = a.as(safe_conversion);
              }
              return af::inverse(src, options);
          },
          py::arg("a").none(false),
          py::arg("options") = af_mat_prop::AF_MAT_NONE,
          "Computes the inverse.\n"
          "If the matrix is an integer matrix, it will be internally casted to either a float or a double "
          "matrix, depending on the capabilities of the current device.");

    m.def("norm",
          [](const af::array &a, const af_norm_type type = af_norm_type::AF_NORM_EUCLID, const double p = 1.0,
             const double q = 1.0) {
              auto src = a;
              if (!a.isfloating()) {
                  auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
                  PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
                  src = a.as(safe_conversion);
              }
              return af::norm(src, type, p, q);
          },
          py::arg("a").none(false),
          py::arg("type") = af_norm_type::AF_NORM_EUCLID,
          py::arg("p") = 1.0,
          py::arg("q") = 1.0);

    m.def("pinverse",
          [](const af::array &a, const double tol = 1e-6) {
              auto src = a;
              if (!a.isfloating()) {
                  auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
                  PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
                  src = a.as(safe_conversion);
              }
              return af::pinverse(src, tol);
          },
          py::arg("a").none(false),
          py::arg("tol") = 1e-6);

    m.def("rank",
          [](const af::array &a, const double tol = 1e-5) {
              auto src = a;
              if (!a.isfloating()) {
                  auto safe_conversion = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
                  PyErr_WarnEx(PyExc_UserWarning, "Automatic conversion to floating array", 1);
                  src = a.as(safe_conversion);
              }
              return af::rank(src, tol);
          },
          py::arg("a").none(false),
          py::arg("tol") = 1e-5);

    m.def("cholesky",
          [](const af::array &a, const bool is_upper = true) {
              af::array out;
              auto errRank = af::cholesky(out, a, is_upper);
              if (errRank != 0) {
                  std::stringstream ss;
                  ss << "Cholesky decomposition failed at rank " << errRank << std::endl;
                  throw std::runtime_error(ss.str());
              }
              return out;
          },
          py::arg("a").none(false),
          py::arg("is_upper") = true,
          "Computes the Cholesky decomposition a positive definite matrix.\n"
          "The resulting matrix is the triangular matrix of the decomposition; multiply it with its "
          "conjugate transpose to reproduce the input matrix.");

    m.def("choleskyInPlace",
          [](af::array &a, const bool is_upper = true) {
              auto errRank = af::choleskyInPlace(a, is_upper);
              if (errRank != 0) {
                  std::stringstream ss;
                  ss << "Cholesky decomposition failed at rank " << errRank << std::endl;
                  throw std::runtime_error(ss.str());
              }
              return a;
          },
          py::arg("a").none(false),
          py::arg("is_upper") = true,
          "Computes the Cholesky decomposition a positive definite matrix.\n"
          "The resulting matrix is the triangular matrix of the decomposition; multiply it with its "
          "conjugate transpose to reproduce the input matrix.");

    m.def("lu",
          [](const af::array &a) {
              af::array lower, upper, pivot;
              af::lu(lower, upper, pivot, a);
              py::tuple result(3);
              result[0] = lower;
              result[1] = upper;
              result[2] = pivot;
              return result;
          },
          py::arg("a").none(false));

    m.def("luInPlace",
          [](af::array &a) {
              af::array pivot;
              af::luInPlace(pivot, a);
              py::tuple result(2);
              result[0] = pivot;
              result[1] = a;
              return result;
          },
          py::arg("a").none(false));


    m.def("qr",
          [](const af::array &a) {
              af::array q, r, tau;
              af::qr(q, r, tau, a);
              py::tuple result(3);
              result[0] = q;
              result[1] = r;
              result[2] = tau;
              return result;
          },
          py::arg("a").none(false));

    m.def("qrInPlace",
          [](af::array &a) {
              af::array tau;
              af::qrInPlace(tau, a);
              py::tuple result(2);
              result[0] = tau;
              result[1] = a;
              return result;
          },
          py::arg("a").none(false));

    m.def("svd",
          [](const af::array &a) {
              af::array u, s, vt;
              af::svd(u, s, vt, a);
              py::tuple result(3);
              result[0] = u;
              result[1] = s;
              result[2] = vt;
              return result;
          },
          py::arg("a").none(false));

    m.def("svdInPlace",
          [](af::array &a) {
              af::array u, s, vt;
              af::svdInPlace(u, s, vt, a);
              py::tuple result(3);
              result[0] = u;
              result[1] = s;
              result[2] = vt;
              return result;
          },
          py::arg("a").none(false));

    m.def("dot",
          [](const af::array &lhs, const af::array &rhs, const bool conj_lhs, const bool conj_rhs) {
              auto lhs_options = conj_lhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              auto rhs_options = conj_rhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              return af::dot(lhs, rhs, lhs_options, rhs_options);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("conj_lhs") = false,
          py::arg("conj_rhs") = false,
          "Scalar dot product between two vectors.  The result is kept in an 1x1 array");

    m.def("dot_scalar",
          [](const af::array &lhs, const af::array &rhs, const bool conj_lhs, const bool conj_rhs) {
              std::variant<double, std::complex<double>> result;
              double real, imag;

              auto lhs_options = conj_lhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              auto rhs_options = conj_rhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              throw_on_error(af_dot_all(&real, &imag, lhs.get(), rhs.get(), lhs_options, rhs_options));
              if (lhs.iscomplex() || rhs.iscomplex())
                  result = std::complex(real, imag);
              else
                  result = real;
              return result;
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("conj_lhs") = false,
          py::arg("conj_rhs") = false,
          "Scalar dot product between two vectors.");

    m.def("gemm",
          [](const af::array &a, const af::array &b, std::optional<af::array> &c,
             const py::float_ &alpha, const py::float_ &beta,
             const bool transA, const bool transB) {

              auto a_options = transA ? af::matProp::AF_MAT_TRANS : af::matProp::AF_MAT_NONE;
              auto b_options = transB ? af::matProp::AF_MAT_TRANS : af::matProp::AF_MAT_NONE;
              auto is32BitsOp = a.type() == af::dtype::f32 || a.type() == af::dtype::c32;

              af_array out = c.has_value() ? c->get() : nullptr;
              af_err err;

              if (is32BitsOp) {
                  auto alpha_f = py::cast<float>(alpha);
                  auto beta_f = c.has_value() ? py::cast<float>(beta) : 0.0f;
                  err = af_gemm(&out, a_options, b_options, &alpha_f, a.get(), b.get(), &beta_f);
              } else {
                  auto alpha_d = py::cast<double>(alpha);
                  auto beta_d = c.has_value() ? py::cast<double>(beta) : 0.0;
                  err = af_gemm(&out, a_options, b_options, &alpha_d, a.get(), b.get(), &beta_d);
              }

              throw_on_error(err);

              if (c.has_value())
                  return c.value();

              return af::array(out);
          },
          py::arg("a").none(false),
          py::arg("b").none(false),
          py::arg("c").none(true) = py::none(),
          py::arg("alpha") = 1.0,
          py::arg("beta") = 0.0,
          py::arg("transA") = false,
          py::arg("transB") = false,
          "Performs a GEMM operation ```C = \\alpha * opA(A)opB(B) + \\beta * C```");

    m.def("matmulTT",
          [](const af::array &lhs, const af::array &rhs) {
              return af::matmulTT(lhs, rhs);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on each one, without taking further memory.");

    m.def("matmulTN",
          [](const af::array &lhs, const af::array &rhs) {
              return af::matmulTN(lhs, rhs);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on lhs, without taking further memory.");

    m.def("matmulNT",
          [](const af::array &lhs, const af::array &rhs) {
              return af::matmulNT(lhs, rhs);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on rhs, without taking further memory.");

    m.def("matmul",
          [](const af::array &lhs, const af::array &rhs, const af::matProp lhs_options, const af::matProp rhs_options) {
              return af::matmul(lhs, rhs, lhs_options, rhs_options);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("lhs_options") = af::matProp::AF_MAT_NONE,
          py::arg("rhs_options") = af::matProp::AF_MAT_NONE,
          "Matrix multiplication with the desired transformations, without taking further memory.");

    m.def("matmul_chain",
          [](const py::args &args) {
              for (auto item: args) {
                  if (!py::isinstance<af::array>(item))
                      throw std::runtime_error("All elements must be matrices");
              }

              auto matrices = args.size();
              if (matrices < 2)
                  throw std::runtime_error("At least two matrices should be present");

              auto first = py::cast<af::array>(args[0]);
              auto acc = af::identity(first.dims(), first.type());

              auto i = 0;
              while (matrices > 0) {
                  if (matrices >= 3) {
                      acc = af::matmul(acc,
                                       py::cast<af::array>(args[i]),
                                       py::cast<af::array>(args[i + 1]),
                                       py::cast<af::array>(args[i + 2]));
                      matrices -= 3;
                      i += 3;
                  } else if (matrices >= 2) {
                      acc = af::matmul(acc,
                                       py::cast<af::array>(args[i]),
                                       py::cast<af::array>(args[i + 1]));
                      matrices -= 2;
                      i += 2;
                  } else if (matrices >= 1) {
                      acc = af::matmul(acc, py::cast<af::array>(args[i]));
                      matrices -= 1;
                      i += 1;
                  }
              }

              return acc;
          }, "Chains matrix multiplications");
}
