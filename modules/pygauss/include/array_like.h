#ifndef AF_ARRAY_LIKE__H
#define AF_ARRAY_LIKE__H

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <type_traits>
#include <complex>
#include "af_array/af_array.h"
#include <boost/type_traits.hpp>

namespace py = pybind11;

namespace array_like {

    namespace detail {

        bool is_af_array(const py::object &value);
        std::optional<af::array> from_af_array(const py::object &value);
        std::optional<af::array> from_scalar(const py::object &value, const af::dim4 &shape);

        bool is_arrow(const py::object &value);
        std::optional<af::array> from_arrow(const py::object &value);

        bool is_numpy(const py::object &value);
        std::optional<af::array> from_numpy(const py::object &value);
    }

    /**
     * Returns true if the object is a scalar, that is, if the
     * object represents either an py::int_, py::float_,
     * py::bool_ or std::complex<double>
     */
    bool is_scalar(const py::object &value);


    /**
     * Returns true if the object can be interpreted as an array by
     * checking internal representations
     */
    bool is_array(const py::object &value);

    /**
     * Returns an af::array instance based on the python object received.
     *
     * This implementation tests first if the actual object itself represents
     * an af::array or a ParallelFor before testing other possible conversions.
     *
     * If shape and value are given, those will be used to ensure the returning
     * array matches the specification.
     *
     * If the value is a scalar and shape is provided (optionally, dtype), a constant
     * array is created matching the shape requirements.  If a scalar is found but
     * no shape is provided, this method will return an empty optional.
     *
     * @return   An empty optional container if the conversion cannot be executed; otherwise,
     * a valid arrayfire instance.
     */
    std::optional<af::array> to_array(const py::object &value,
                                      const std::optional<af::dim4> &shape = std::nullopt,
                                      const std::optional<af::dtype> &dtype = std::nullopt);
}

#endif // AF_ARRAY_LIKE__H
