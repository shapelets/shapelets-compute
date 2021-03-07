#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pygauss.h"

namespace py = pybind11;

typedef struct {
} scoped_batch;

void pygauss::bindings::batch_api(py::module &m) {
    py::class_<ParallelFor> pf(m, "ParallelFor");

    pf.def("__iter__",
           [](const ParallelFor &self) {
               return self;
           });

    pf.def("__next__",
           [](const ParallelFor &self) {
               if (GForStatus::get()) {
                   GForStatus::toggle();
                   throw py::stop_iteration();
               }

               GForStatus::toggle();
               return self;
           });

    py::class_<scoped_batch>(m, "ScopedBatch")
            .def("__enter__",
                 [](const scoped_batch &self) {
                     GForStatus::set(true);
                 })
            .def("__exit__",
                 [](const scoped_batch &self, const py::args &) {
                     GForStatus::set(false);
                 });

    m.def("batch", []() { return scoped_batch{}; });

    m.def("batch", [](const py::function &usrFn) {
        auto block = gfor_acquire();
        return usrFn();
    });

    m.def("parallel_range",
          [](const std::variant<py::int_, py::slice> &arg) {
              if (auto p = std::get_if<py::int_>(&arg)) {
                  return ParallelFor((ssize_t) *p);
              }

              auto ps = std::get<py::slice>(arg);
              auto s = pygauss::Slice::from_python(ps);
              return pygauss::ParallelFor(s);
          },
          py::arg("arg").none(false));

}
