#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include <af_array/af_array.h>

namespace py = pybind11;
namespace spd = spdlog;

typedef af_err (*binaryFn)(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

af::array binary_function(const af::array &self, const py::object &other, bool reverse, binaryFn fn) {
    af_array rhs = py::isinstance<af::array>(other) ?
                   py::cast<af::array>(other).get() :
                   constant_array(other, self.dims());

    af_array out = nullptr;
    if (!reverse)
        check_af_error((*fn)(&out, self.get(), rhs, GForStatus::get()));
    else
        check_af_error((*fn)(&out, rhs, self.get(), GForStatus::get()));

    return af::array(out);
}


#define BINARY_OP(OP, PYTHON_FN)                                                                        \
    ka.def(#PYTHON_FN,                                                                                  \
                   [](const af::array &self, const py::object &other){                                  \
                       spd::debug("Binary operation {}", #OP);                                          \
                       return binary_function(self, other, false, OP);                                  \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \


#define BINARY_OPR(OP, PYTHONFN)                                                                        \
    ka.def(#PYTHONFN,                                                                                   \
                   [](const af::array &self, const py::object &other){                                  \
                       spd::debug("Binary reverse operation {}", #OP);                                  \
                       return binary_function(self, other, true, OP);                                   \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \


#define BINARY_IOP(OP, PYTHON_FN)                                                                       \
    ka.def(#PYTHON_FN,                                                                                  \
                   [](af::array &self, const py::object &other){                                        \
                       spd::debug("Binary inplace operation {}", #OP);                                  \
                       self = binary_function(self, other, false, OP);                                  \
                       return self;                                                                     \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \
                                                                                                        \


int array_priority = 30;

void array_obj_bindings(py::module &m) {

    py::class_<af::array> ka(m, "ShapeletsArray", py::buffer_protocol());

    ka.def_buffer(&af_buffer_protocol);

//    ka.def("__array__",
//           [](const af::array &self){
//               return py::memoryview(af_buffer_protocol(self));
//           });

//    ka.def("interpret_slice",
//           [](const af::array &self, const py::slice& slice){
//               auto [start, end, step] = interpret_slice(slice);
//               py::tuple result(3);
//               result[0] = start;
//               result[1] = end;
//               result[2] = step;
//               return result;
//           },
//           py::arg("slice").none(false));

    ka.def("same_as",
           [](const af::array &self, const py::object &arr_like, const py::float_ eps) {

               auto other = py::isinstance<af::array>(arr_like) ?
                            py::cast<af::array>(arr_like) :
                            af_from_array_like(arr_like, std::nullopt, self.type());

               if (self.dims() != other.dims())
                   return false;

               auto typed = other;
               if (other.type() != self.type())
                   typed = other.as(self.type());

               auto non_nan_self = self.copy();
               auto non_nan_other = other.copy();

               non_nan_self(af::where(af::isNaN(self))) = 0.0;
               non_nan_other(af::where(af::isNaN(other))) = 0.0;

               return af::allTrue<bool>(af::abs(non_nan_self - non_nan_other) < (double) eps);
           },
           py::arg("arr_like").none(false),
           py::arg("eps") = 1e-4,
           "Performs a element wise comparison between the arrays and returns "
           "True if the two arrays are the same (same dimensions, same values).");

    ka.def_readonly_static("__array_priority__", &array_priority,
                           "Ensure priority in resolved in favour of our built-in methods");

    ka.def("__getitem__",
           [](const af::array &self, const py::object &selector) {
               af_array out = nullptr;
               auto[res_dim, index_dim, index] = build_index(selector, self.dims());
               check_af_error(af_index_gen(&out, self.get(), res_dim, index));
               check_af_error(af_release_indexers(index));
               return af::array(out);
           },
           py::arg("selector").none(false),
           "");

    ka.def("__setitem__",
           [](af::array &self, const py::object &selector, const py::object &value) {
               auto value_is_array = py::isinstance<af::array>(value);
               auto[res_dim, index_dim, index] = build_index(selector, self.dims());

               af_array rhs = nullptr;
               if (value_is_array) {
                   auto rhsa = py::cast<af::array>(value);
                   if (rhsa.dims() != index_dim) {
                       std::ostringstream msg;
                       msg << "Not the same dimensions: expected " << index_dim << " vs given " << rhsa.dims();
                       throw std::runtime_error(msg.str());
                   }
                   rhs = py::cast<af::array>(value).get();
               }
               else {
                   spd::debug("Creating constant array for __setitem__ operation with dimensions {}, {}, {}, {}",
                              index_dim[0], index_dim[1], index_dim[2], index_dim[3]);
                   rhs = constant_array(value, index_dim);
               }

               af_array out = nullptr;
               auto err = af_assign_gen(&out, self.get(), res_dim, index, rhs);
               spd::debug("The error code for af_assign_gen is {}", err);

               // ensure rhs is removed before checking for errors...
               if (!value_is_array) check_af_error(af_release_array(rhs));
               check_af_error(err);
               check_af_error(af_release_indexers(index));
               self = af::array(out);
               return self;
           },
           py::arg("selector").none(false),
           py::arg("value").none(false));

    ka.def("__copy__",
           [](const af::array &self) {
               return af::array(self);
           }, "Shallow copy");

    ka.def("__deepcopy__",
           [](const af::array &self) {
               return self.copy();
           }, "Deep copy");

    ka.def_property_readonly("backend",
                             [](const af::array &self) {
                                 af_backend result;
                                 check_af_error(af_get_backend_id(&result, self.get()));
                                 return static_cast<af::Backend>(result);
                             });

    ka.def_property_readonly("is_column",
                             [](const af::array &self) {
                                 return self.iscolumn();
                             });

    ka.def_property_readonly("is_row",
                             [](const af::array &self) {
                                 return self.isrow();
                             });

    ka.def_property_readonly("shape",
                             [](const af::array &self) {
                                 return self.dims();
                             });

    ka.def_property_readonly("size",
                             [](const af::array &self) {
                                 return self.elements();
                             }, "Returns the total number of elements of this array.");


    ka.def_property_readonly("ndim",
                             [](const af::array &self) {
                                 return self.numdims();
                             }, "Returns the number of dimensions.");

    ka.def_property_readonly("itemsize",
                             [](const af::array &self) {
                                 return scalar_size(self.type());
                             }, "Returns the size in bytes of each individual item held by this array");

    ka.def_property_readonly("dtype",
                             [](const af::array &self) {
                                 return self.type();
                             }, "Returns the numpy dtype describing the type of elements held by this array");

    ka.def_property_readonly("T",
                             [](const af::array &self) {
                                 return self.T();
                             }, "Get the transposed the array");

    ka.def_property_readonly("H",
                             [](const af::array &self) {
                                 return self.H();
                             }, "Get the conjugate-transpose of the current array");


    ka.def("astype",
           [](const af::array &self, const af::dtype &type) {
               return self.as(type);
           },
           py::arg("type").none(false),
           "converts the array into a new array with the specified type");


    ka.def("__repr__",
           [](const af::array &self) {
               char *out = nullptr;
               af_array_to_string(&out, "", self.get(), 4, true);
               return std::string(out);
           });

    ka.def("display",
           [](const af::array &self, int precision, bool transpose) {
               self.eval();
               char *out = nullptr;
               af_array_to_string(&out, "", self.get(), precision, transpose);
               py::print(py::str(out));
           },
           py::arg("precision") = 4,
           py::arg("transpose") = true);

    BINARY_OP(af_add, __add__)
    BINARY_OPR(af_add, __radd__)
    BINARY_IOP(af_add, __iadd__)

    BINARY_OP(af_sub, __sub__)
    BINARY_OPR(af_sub, __rsub__)
    BINARY_IOP(af_sub, __isub__)

    BINARY_OP(af_mul, __mul__)
    BINARY_OPR(af_mul, __rmul__)
    BINARY_IOP(af_mul, __imul__)

    BINARY_OP(af_div, __truediv__)
    BINARY_OPR(af_div, __rtruediv__)
    BINARY_IOP(af_div, __itruediv__)

    BINARY_OP(af_mod, __mod__)
    BINARY_OPR(af_mod, __rmod__)
    BINARY_IOP(af_mod, __imod__)

    BINARY_OP(af_pow, __pow__)
    BINARY_OPR(af_pow, __rpow__)
    BINARY_IOP(af_pow, __ipow__)

    BINARY_OP(af_eq, __eq__)
    BINARY_OP(af_neq, __ne__)
    BINARY_OP(af_lt, __lt__)
    BINARY_OP(af_le, __le__)
    BINARY_OP(af_ge, __ge__)
    BINARY_OP(af_gt, __gt__)

    BINARY_OP(af_bitand, __and__)
    BINARY_OPR(af_bitand, __rand__)
    BINARY_IOP(af_bitand, __iand__)

    BINARY_OP(af_bitor, __or__)
    BINARY_OPR(af_bitor, __ror__)
    BINARY_IOP(af_bitor, __ior__)

    BINARY_OP(af_bitxor, __xor__)
    BINARY_OPR(af_bitxor, __rxor__)
    BINARY_IOP(af_bitxor, __ixor__)


    BINARY_OP(af_bitshiftl, __lshift__)
    BINARY_OPR(af_bitshiftl, __rlshift__)
    BINARY_IOP(af_bitshiftl, __ilshift__)

    BINARY_OP(af_bitshiftr, __rshift__)
    BINARY_OPR(af_bitshiftr, __rrshift__)
    BINARY_IOP(af_bitshiftr, __irshift__)

    ka.def("__neg__",
           [](const af::array &self, const py::object &other) {
               auto zero = py::float_(0.0);
               return binary_function(self, zero, true, af_sub);
           }, py::arg("other").none(false));

    m.def("eval",
          [](const py::args &args) {
              if (args.is_none() || args.empty())
                  return;
              std::vector<af_array> lst;
              for (size_t i = 0; i < args.size(); i++) {
                  auto obj = args[i];
                  if (!obj.is_none() && py::isinstance<af::array>(obj))
                      lst.push_back(py::cast<af::array>(obj).get());
              }
              if (!lst.empty()) {
                  check_af_error(af_eval_multiple(lst.size(), lst.data()));
              }
          },
          "Forces the evaluation of all the arrays");
}
