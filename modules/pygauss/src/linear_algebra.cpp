#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pygauss.h>

namespace py = pybind11;

void pygauss::bindings::linear_algebra_operations(py::module &m) {



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

    m.def("det",
          [](const py::object &array_like) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              std::variant<py::float_, std::complex<double>> result;

              if (a.iscomplex()) {
                  double real, img = 0.0;
                  throw_on_error(af_det(&real, &img, a.get()));
                  result = std::complex<double>(real, img);
              } else {
                  if (a.type() == af::dtype::f64)
                      result = py::float_(af::det<double>(a));
                  else
                      result = py::float_(af::det<float>(a));
              }

              return result;
          }, py::arg("array_like").none(false),
          "Computes the determinant of a matrix.");

    m.def("inverse",
          [](const py::object &array_like, const af_mat_prop options = af_mat_prop::AF_MAT_NONE) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              return af::inverse(a, options);
          },
          py::arg("array_like").none(false),
          py::arg("options") = af_mat_prop::AF_MAT_NONE,
          "Computes the inverse.");

    m.def("norm",
          [](const py::object &array_like, const af_norm_type type, const double p, const double q) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              return af::norm(a, type, p, q);
          },
          py::arg("array_like").none(false),
          py::arg("type") = af_norm_type::AF_NORM_EUCLID,
          py::arg("p") = 1.0,
          py::arg("q") = 1.0);

    m.def("pinverse",
          [](const py::object &array_like, const double tol) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              return af::pinverse(a, tol);
          },
          py::arg("array_like").none(false),
          py::arg("tol") = 1e-6);

    m.def("rank",
          [](const py::object &array_like, const double tol) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              return af::rank(a, tol);
          },
          py::arg("array_like").none(false),
          py::arg("tol") = 1e-5);

    m.def("cholesky",
          [](const py::object &array_like, const bool is_upper) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              af::array out;
              auto errRank = af::cholesky(out, a, is_upper);
              if (errRank != 0) {
                  std::stringstream ss;
                  ss << "Cholesky decomposition failed at rank " << errRank << std::endl;
                  throw std::runtime_error(ss.str());
              }
              return out;
          },
          py::arg("array_like").none(false),
          py::arg("is_upper") = true,
          "Computes the Cholesky decomposition a positive definite matrix.\n"
          "The resulting matrix is the triangular matrix of the decomposition; multiply it with its "
          "conjugate transpose to reproduce the input matrix.");

    m.def("lu",
          [](const py::object &array_like) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              af::array lower, upper, pivot;
              af::lu(lower, upper, pivot, a);
              py::tuple result(3);
              result[0] = lower;
              result[1] = upper;
              result[2] = pivot;
              return result;
          },
          py::arg("array_like").none(false));

    m.def("qr",
          [](const py::object &array_like) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              af::array q, r, tau;
              af::qr(q, r, tau, a);
              py::tuple result(3);
              result[0] = q;
              result[1] = r;
              result[2] = tau;
              return result;
          },
          py::arg("array_like").none(false));

    m.def("svd",
          [](const py::object &array_like) {
              auto a = arraylike::as_array_checked(array_like);
              arraylike::ensure_floating(a);

              af::array u, s, vt;
              af::svd(u, s, vt, a);
              py::tuple result(3);
              result[0] = u;
              result[1] = s;
              result[2] = vt;
              return result;
          },
          py::arg("array_like").none(false));

    m.def("dot",
          [](const py::object &lhs, const py::object &rhs, const bool conj_lhs, const bool conj_rhs) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);
              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              auto lhs_options = conj_lhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              auto rhs_options = conj_rhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              return af::dot(l, r, lhs_options, rhs_options);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("conj_lhs") = false,
          py::arg("conj_rhs") = false,
          "Scalar dot product between two vectors.  Also referred to as the inner product.  The "
          "result is kept as an array in the device.");

    m.def("dot_scalar",
          [](const py::object &lhs, const py::object &rhs, const bool conj_lhs, const bool conj_rhs) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);

              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              auto lhs_options = conj_lhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;
              auto rhs_options = conj_rhs ? af::matProp::AF_MAT_CONJ : af::matProp::AF_MAT_NONE;

              std::variant<double, std::complex<double>> result;
              double real, imag;
              throw_on_error(af_dot_all(&real, &imag, l.get(), r.get(), lhs_options, rhs_options));
              if (l.iscomplex() || r.iscomplex())
                  result = std::complex(real, imag);
              else
                  result = real;
              return result;
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("conj_lhs") = false,
          py::arg("conj_rhs") = false,
          "Scalar dot product between two vectors.  Also referred to as the inner product.  The "
          "result returned as a scalar.");

    m.def("gemm",
          [](const py::object &a, const py::object &b, const py::object &c,
             const py::float_ &alpha, const py::float_ &beta, const bool transA, const bool transB) {

              auto a_options = transA ? af::matProp::AF_MAT_TRANS : af::matProp::AF_MAT_NONE;
              auto b_options = transB ? af::matProp::AF_MAT_TRANS : af::matProp::AF_MAT_NONE;

              // A drives the operation...
              auto A = arraylike::as_array_checked(a);
              auto is32BitsOp = A.type() == af::dtype::f32 || A.type() == af::dtype::c32;

              // Ensure B matches the type of A
              auto B = arraylike::as_array_checked(b);
              if (B.type() != A.type())
                  B = B.as(A.type());

              // C is optional, but if present, it is the result
              auto C = arraylike::as_array(c);

              af_array out = nullptr;
              // If it has value, ensure it matches the type of A
              if (C.has_value()) {
                  if (C->type() != A.type())
                      // do not loose the conversion otherwise it will be gc'd
                      C = C->as(A.type());
                  // Get the handle
                  out = C->get();
              }

              if (is32BitsOp) {
                  // Adjust alpha and beta to floats
                  auto alpha_f = py::cast<float>(alpha);
                  auto beta_f = C.has_value() ? py::cast<float>(beta) : 0.0f;
                  throw_on_error(af_gemm(&out, a_options, b_options, &alpha_f, A.get(), B.get(), &beta_f));
              } else {
                  // adjust alpha and beta to doubles
                  auto alpha_d = py::cast<double>(alpha);
                  auto beta_d = C.has_value() ? py::cast<double>(beta) : 0.0;
                  throw_on_error(af_gemm(&out, a_options, b_options, &alpha_d, A.get(), B.get(), &beta_d));
              }

              // If C was present, it also has the value (we used the af_array of C for the computation)
              // return it.
              if (C.has_value())
                  return C.value();

              // Otherwise, the result should be in out.
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
          [](const py::object &lhs, const py::object &rhs) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);
              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              return af::matmulTT(l, r);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on each one, without taking further memory.");

    m.def("matmulTN",
          [](const py::object &lhs, const py::object &rhs) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);
              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              return af::matmulTN(l, r);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on lhs, without taking further memory.");

    m.def("matmulNT",
          [](const py::object &lhs, const py::object &rhs) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);
              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              return af::matmulNT(l, r);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          "Matrix multiplication after performing a transpose on rhs, without taking further memory.");

    m.def("matmul",
          [](const py::object &lhs, const py::object &rhs, const af::matProp lhs_options,
             const af::matProp rhs_options) {
              auto l = arraylike::as_array_checked(lhs);
              auto r = arraylike::as_array_checked(rhs);
              arraylike::ensure_floating(l);
              arraylike::ensure_floating(r);

              return af::matmul(l, r, lhs_options, rhs_options);
          },
          py::arg("lhs").none(false),
          py::arg("rhs").none(false),
          py::arg("lhs_options") = af::matProp::AF_MAT_NONE,
          py::arg("rhs_options") = af::matProp::AF_MAT_NONE,
          "Matrix multiplication with the desired transformations, without taking further memory.");

    m.def("matmul_chain",
          [](const py::args &args) {

              auto matrices = args.size();
              if (matrices < 2)
                  throw std::runtime_error("At least two matrices should be present");

              auto checked = std::vector<af::array>(args.size());

              for (auto item: args) {
                  auto obj = py::cast<py::object>(item);
                  auto arr = arraylike::as_array_checked(py::cast<py::object>(obj));
                  arraylike::ensure_floating(arr);
                  // The final check is to ensure the arrays are all floating point.
                  checked.push_back(arr);
              }

              auto first = checked[0];
              auto acc = af::identity(first.dims(), first.type());

              auto i = 0;
              while (matrices > 0) {
                  if (matrices >= 3) {
                      acc = af::matmul(acc, checked[i], checked[i+1], checked[i+2]);
                      matrices -= 3;
                      i += 3;
                  } else if (matrices >= 2) {
                      acc = af::matmul(acc, checked[i], checked[i+1]);
                      matrices -= 2;
                      i += 2;
                  } else if (matrices >= 1) {
                      acc = af::matmul(acc, checked[i]);
                      matrices -= 1;
                      i += 1;
                  }
              }

              return acc;
          }, "Chains matrix multiplications");

    // These in-place functions may move to the actual object itself, but I am a little bit unsure

//    m.def("choleskyInPlace",
//          [](const py::object &array_like, const bool is_upper = true) {
//              auto errRank = af::choleskyInPlace(a, is_upper);
//              if (errRank != 0) {
//                  std::stringstream ss;
//                  ss << "Cholesky decomposition failed at rank " << errRank << std::endl;
//                  throw std::runtime_error(ss.str());
//              }
//              return a;
//          },
//          py::arg("a").none(false),
//          py::arg("is_upper") = true,
//          "Computes the Cholesky decomposition a positive definite matrix.\n"
//          "The resulting matrix is the triangular matrix of the decomposition; multiply it with its "
//          "conjugate transpose to reproduce the input matrix.");

//    m.def("luInPlace",
//          [](af::array &a) {
//              af::array pivot;
//              af::luInPlace(pivot, a);
//              py::tuple result(2);
//              result[0] = pivot;
//              result[1] = a;
//              return result;
//          },
//          py::arg("a").none(false));

//    m.def("qrInPlace",
//          [](af::array &a) {
//              af::array tau;
//              af::qrInPlace(tau, a);
//              py::tuple result(2);
//              result[0] = tau;
//              result[1] = a;
//              return result;
//          },
//          py::arg("a").none(false));

//    m.def("svdInPlace",
//          [](af::array &a) {
//              af::array u, s, vt;
//              af::svdInPlace(u, s, vt, a);
//              py::tuple result(3);
//              result[0] = u;
//              result[1] = s;
//              result[2] = vt;
//              return result;
//          },
//          py::arg("a").none(false));

}
