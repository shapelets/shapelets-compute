#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include <pygauss.h>

namespace py = pybind11;

void pygauss::bindings::array_construction_operations(py::module &m) {

    m.def("empty",
          [](const af::dim4 &shape, const af::dtype &dtype) {
              return af::array(shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32
    );

    m.def("eye",
          [](const int N, const std::optional<int>& M, const int k, const af::dtype &dtype) {
              if (N <= 0)
                  throw std::runtime_error("N cannot be negative or zero");

              auto n = N;
              auto m = M.value_or(N);

              auto one_count = std::min(n,m) - std::abs(k);
              auto ones = af::constant(1, one_count, dtype);
              auto result = af::diag(ones, k, false);
              auto result_rows = result.dims(0);
              auto result_cols = result.dims(1);
              if (result_rows < n || result_cols < m) {
                  auto end_padding = af::dim4(n-result_rows, m-result_cols,0,0);
                  result = af::pad(result, af::dim4(0,0,0,0), end_padding, af::borderType::AF_PAD_ZERO);
              }
              return result;
          },
          py::arg("N").none(false),
          py::arg("M") = py::none(),
          py::arg("k") = 0,
          py::arg("dtype") = af::dtype::f32,
          R"_(
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : array of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.
)_");

    m.def("identity",
          [](const af::dim4 &shape, const af::dtype &dtype) {
              return af::identity(shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          R"_(
    Creates an identity array with diagonal values set to one.

    Parameters
    ----------
    shape : Shape
        When set to an integer, n, it will return return an identity matrix
        of nxn.  Otherwise, use a tuple to specify the exact dimensions for
        the tensor where all the diagonal elements will be set to 1.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : Array
        Array of shape ``shape`` whose diagonal elements are all set to zero.
)_");

    m.def("full",
          [](const af::dim4 &shape, const py::object &fill_value, const af::dtype &dtype) {
              auto candidate = arraylike::scalar_as_array(fill_value, shape);
              if (!candidate)
                  throw std::invalid_argument("Unable to create array");
              auto result = candidate.value();
              if (result.type() == dtype)
                  return result;

              return result.as(dtype);
          },
          py::arg("shape").none(false),
          py::arg("fill_value").none(false),
          py::arg("dtype") = af::dtype::f32,
          R"_(
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  If not specified it will
        default to "float".
)_");

    m.def("zeros",
          [](const af::dim4 &shape, const af::dtype &dtype) {
              return af::constant(0, shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          "Creates an array with the given dimensions with all its elements set to zero.");

    m.def("ones",
          [](const af::dim4 &shape, const af::dtype &dtype) {
              return af::constant(1, shape, dtype);
          },
          py::arg("shape").none(false),
          py::arg("dtype") = af::dtype::f32,
          "Creates an array with the given dimensions with all its elements set to zero.");

    m.def("diag",
          [](const py::object &arr_like, int index = 0, bool extract = false) {
              auto a = arraylike::as_array_checked(arr_like);
              return af::diag(a, index, extract);
          },
          py::arg("a").none(false),
          py::arg("index") = 0,
          py::arg("extract") = false,
          "Operates with diagonals\n"
          "Using extract parameter one is able to either create a diagonal matrix from a vector (false) or "
          "extract a diagonal from a matrix to a vector (true)");

    m.def("array",
          [](const py::object& array_like,
             const std::optional<af::dim4> &shape,
             const std::optional<af::dtype> &dtype) {

              return arraylike::as_array_like(array_like, shape, dtype);
          },
          py::arg("array_like").none(false),
          py::arg("shape") = py::none(),
          py::arg("dtype") = py::none(),
          R"_(
    Converts and interprets the input as an array or tensor.

    Possible inputs are native Python constructs like lists and tuples, but also numpy arrays or
    arrow constructs.  Basically, it will process any object that has array semantics either through
    array methods or buffer protocols.

    Parameters
    ----------
    array_like: ArrayLike construct
    shape: Int or Tuple of ints. Defaults to None
         When shape is set, array_like object will be adjusted to match the given dimensionality.
    dtype: A compatible expression numpy dtype.
         When dtype is not set, the type will be inferred from the actual array_like object

    Examples
    --------
    Create a two dimensional array:

    >>> import shapelets.compute as sh
    >>> a = sh.array([[1,2],[3,4]])

    )_");

    m.def("iota",
          [](const af::dim4 &shape, const af::dim4 &tile, const af::dtype &dtype) {
              return af::iota(shape, tile, dtype);
          },
          py::arg("shape").none(false),
          py::arg("tile") = af::dim4(1),
          py::arg("dtype") = af::dtype::f32,
          "Create an sequence [0, shape.elements() - 1] and modify to specified dimensions dims "
          "and then tile it according to tile");

    m.def("range", [](const af::dim4 &shape, int seq_dim, const af::dtype &dtype) {
              return af::range(shape, seq_dim, dtype);
          },
          py::arg("shape").none(false),
          py::arg("seq_dim") = -1,
          py::arg("dtype") = af::dtype::f32,
          "Creates an array with [0, n] values along the seq_dim which is tiled across other dimensions.");

}
