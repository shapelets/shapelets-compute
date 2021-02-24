#ifndef KHIVA_NUMPY__H
#define KHIVA_NUMPY__H

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Exposes an af array through the buffer protocol
 * @param self
 * @return
 */
py::buffer_info af_buffer_protocol(const af::array &self);

/**
 * Builds an af array from a python construct that resembles an array
 * @param arr_like
 * @param shape
 * @param dtype
 * @return
 */
af::array af_from_array_like(const py::object &arr_like,
                             const std::optional<af::dim4> &shape,
                             const std::optional<af::dtype> &dtype);




#endif // KHIVA_NUMPY__H