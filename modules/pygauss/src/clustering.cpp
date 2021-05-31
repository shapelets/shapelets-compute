/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pygauss.h>

namespace py = pybind11;

void pygauss::bindings::clustering_functions(py::module &m) {

    m.def(
        "kshape_classify",
        [](const py::object &data, const py::object &obj_centroids) {
            auto tss = arraylike::as_array_checked(data);
            auto centroids = arraylike::as_array_checked(obj_centroids);

            arraylike::ensure_floating(tss);
            arraylike::ensure_floating(centroids);

            return gauss::clustering::kshape_classify(tss, centroids);
        },
        py::arg("data").none(false),
        py::arg("obj_centroids").none(false));

    m.def(
        "kshape_calibrate",
        [](const py::object &data, const int k, const py::object &labels, const int max_iterations, const bool rnd_labels) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);

            af::array lbls;
            af::array centroids;

            if (!labels.is_none())
                lbls = arraylike::as_array_checked(labels);
            
            gauss::clustering::kshape_calibrate(tss, k, centroids, lbls, max_iterations, rnd_labels);
            return py::make_tuple(lbls, centroids);
        },
        py::arg("tss").none(false),
        py::arg("k").none(false),
        py::arg("labels") = py::none(),
        py::arg("max_iterations") = 100,
        py::arg("rnd_labels") = false);
}