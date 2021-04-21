#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

#include <pygauss.h>

namespace py = pybind11;
namespace spd = spdlog;

typedef af_err (*binaryFn)(af_array *out, const af_array lhs, const af_array rhs, const bool batch);

af::array binary_function(af::array &self, const py::object &other, bool reverse, binaryFn fn) {
    af::array rhs = pygauss::arraylike::as_itself_or_promote(other, self.dims(), self.type());
    auto batch = pygauss::GForStatus::get() || self.dims() != rhs.dims();
    af_array out = nullptr;
    if (!reverse)
        pygauss::throw_on_error((*fn)(&out, self.get(), rhs.get(), batch));
    else
        pygauss::throw_on_error((*fn)(&out, rhs.get(), self.get(), batch));

    return af::array(out);
}

#define BINARY_OP(OP, PYTHON_FN)                                                                        \
    ka.def(#PYTHON_FN,                                                                                  \
                   [](af::array &self, const py::object &other){                                        \
                       spd::trace("Binary operation {} {}", #OP, GForStatus::get());                    \
                       return binary_function(self, other, false, OP);                                  \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \


#define BINARY_OPR(OP, PYTHON_FN)                                                                       \
    ka.def(#PYTHON_FN,                                                                                  \
                   [](af::array &self, const py::object &other){                                        \
                       spd::trace("Binary operation {} {}", #OP, GForStatus::get());                    \
                       return binary_function(self, other, true, OP);                                   \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \


#define BINARY_IOP(OP, PYTHON_FN)                                                                       \
    ka.def(#PYTHON_FN,                                                                                  \
                   [](af::array &self, const py::object &other){                                        \
                       spd::trace("Binary operation {} {}", #OP, GForStatus::get());                    \
                       self = binary_function(self, other, false, OP);                                  \
                       return self;                                                                     \
                   },                                                                                   \
                   py::arg("other").none(false));                                                       \
                                                                                                        \


int array_priority = 30;

void pygauss::bindings::array_obj(py::module &m) {

    py::class_<af::array> ka(m, "ShapeletsArray", py::buffer_protocol());

    ka.def_buffer(&pygauss::arraylike::numpy_buffer_protocol);

    ka.def("same_as",
           [](const af::array &self, const py::object &arr_like, const py::float_ &eps) {
               auto other = pygauss::arraylike::as_array_like(arr_like, self);
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
               auto[res_dim, index_dim, index] = pygauss::arraylike::build_index(selector, self.dims());
               throw_on_error(af_index_gen(&out, self.get(), res_dim, index));
               throw_on_error(af_release_indexers(index));
               return af::array(out);
           },
           py::arg("selector").none(false),
           "");

    ka.def("__setitem__",
           [](af::array &self, const py::object &selector, const py::object &value) {
               // Interpret the index expression...
               auto[res_dim, index_dim, index] = pygauss::arraylike::build_index(selector, self.dims());
               af::array rhs = arraylike::as_itself_or_promote(value, index_dim, self.type());
               if (rhs.type() != self.type())
                   rhs = rhs.as(self.type());

               // do the assigment...
               af_array out = nullptr;
               auto err = af_assign_gen(&out, self.get(), self.numdims(), index, rhs.get());
               throw_on_error(af_release_indexers(index));
               throw_on_error(err);
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
           [](const af::array &self, const py::object& memo) {
               return self.copy();
           }, py::arg("memo"), "Deep copy");

    ka.def("__len__",
           [](const af::array &self) {
               auto dims = self.dims();
               return dims[0];
           });       

    ka.def_property_readonly("backend",
                             [](const af::array &self) {
                                 af_backend result;
                                 throw_on_error(af_get_backend_id(&result, self.get()));
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

    ka.def_property_readonly("is_vector",
                             [](const af::array &self) {
                                 return self.isvector();
                             });

    ka.def_property_readonly("is_empty",
                             [](const af::array &self) { 
                                 return self.isempty();
                             });                         

    ka.def_property_readonly("is_single",
                             [](const af::array &self) {
                                 return self.issingle();
                             });

    ka.def_property_readonly("is_integer",
                             [](const af::array &self) {
                                 return self.isinteger();
                             });

    ka.def_property_readonly("is_complex",
                             [](const af::array &self) {
                                 return self.iscomplex();
                             });

    ka.def_property_readonly("is_bool",
                             [](const af::array &self) {
                                 return self.isbool();
                             });

    ka.def_property_readonly("is_floating",
                             [](const af::array &self) {
                                 return self.isfloating();
                             });

    ka.def_property_readonly("is_half",
                             [](const af::array &self) {
                                 return self.ishalf();
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
                                 return arraylike::scalar_size(self.type());
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

    ka.def_property_readonly("real",
                             [](const af::array &self) {
                                return af::real(self);
                             }, "Returns the real part of this tensor");

    ka.def_property_readonly("imag",
                             [](const af::array &self) {
                                 return af::imag(self);
                             }, "Returns the imaginary part of this tensor");

    ka.def("eval",
           [](const af::array &self){
               self.eval();
           });
    
    ka.def("astype",
           [](const af::array &self, const af::dtype &type) {
               return self.as(type);
           },
           py::arg("type").none(false),
           "converts the array into a new array with the specified type");

    ka.def("__int__", 
        [](const af::array &self){
            
            if (self.elements() != 1)
                throw std::runtime_error("Only arrays of one element can be converted to Python scalars");
            switch (self.type())
            {
            case af::dtype::b8: {
                auto d = self.scalar<char>();
                return py::int_(d);
            }
            case af::dtype::c32: {
                auto d = self.scalar<af::af_cfloat>();
                return py::int_(static_cast<long long>(d.real));
            }
            case af::dtype::c64: {
                auto d = self.scalar<af::af_cdouble>();
                return py::int_(static_cast<long long>(d.real));
            }
            case af::dtype::s16: {
                auto d = self.scalar<short>();
                return py::int_(d);
            }
            case af::dtype::s32: {
                auto d = self.scalar<int>();
                return py::int_(d);
            }
            case af::dtype::s64: {
                auto d = self.scalar<long long>();
                return py::int_(d);
            }
            case af::dtype::f16: {
                auto d = self.scalar<af::half>();
                auto f = static_cast<float>(half::from_bits(d.data_));
                return py::int_(static_cast<int>(f));
            }
            case af::dtype::f32: {
                auto d = self.scalar<float>();
                return py::int_(static_cast<long long>(d));
            }
            case af::dtype::f64: {
                auto d = self.scalar<double>();
                return py::int_(static_cast<long long>(d));
            }
            case af::dtype::u16: {
                auto d = self.scalar<unsigned short>();
                return py::int_(d);
            }
            case af::dtype::u32: {
                auto d = self.scalar<unsigned int>();
                return py::int_(d);
            }
            case af::dtype::u64: {
                auto d = self.scalar<unsigned long long>();
                return py::int_(d);
            }
            case af::dtype::u8: {
                auto d = self.scalar<unsigned char>();
                return py::int_(d);
            }
            default:
                throw std::runtime_error("Unexpected type");
                break;
            }                  
        });

    ka.def("__float__", 
        [](const af::array &self){
            if (self.elements() != 1)
                throw std::runtime_error("Only arrays of one element can be converted to Python scalars");

            switch (self.type())
            {
            case af::dtype::b8: {
                auto d = self.scalar<char>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::c32: {
                auto d = self.scalar<af::af_cfloat>();
                return py::float_(d.real);
            }
            case af::dtype::c64: {
                auto d = self.scalar<af::af_cdouble>();
                return py::float_(d.real);
            }
            case af::dtype::s16: {
                auto d = self.scalar<short>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::s32: {
                auto d = self.scalar<int>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::s64: {
                auto d = self.scalar<long long>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::f16: {
                auto d = self.scalar<af::half>();
                auto f = static_cast<float>(half::from_bits(d.data_));
                return py::float_(f);
            }
            case af::dtype::f32: {
                auto d = self.scalar<float>();
                return py::float_(d);
            }
            case af::dtype::f64: {
                auto d = self.scalar<double>();
                return py::float_(d);
            }
            case af::dtype::u16: {
                auto d = self.scalar<unsigned short>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::u32: {
                auto d = self.scalar<unsigned int>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::u64: {
                auto d = self.scalar<unsigned long long>();
                return py::float_(static_cast<double>(d));
            }
            case af::dtype::u8: {
                auto d = self.scalar<unsigned char>();
                return py::float_(static_cast<double>(d));
            }
            default:
                throw std::runtime_error("Unexpected type");
                break;
            }                
        });

    ka.def("__complex__", 
        [](const af::array &self){
            if (self.elements() != 1)
                throw std::runtime_error("Only arrays of one element can be converted to Python scalars");

            switch (self.type())
            {
            case af::dtype::b8: {
                auto d = self.scalar<char>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::c32: {
                auto d = self.scalar<af::af_cfloat>();
                return std::complex<double>(static_cast<double>(d.real), static_cast<double>(d.imag));
            }
            case af::dtype::c64: {
                auto d = self.scalar<af::af_cdouble>();
                return std::complex<double>(d.real, d.imag);
            }
            case af::dtype::s16: {
                auto d = self.scalar<short>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::s32: {
                auto d = self.scalar<int>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::s64: {
                auto d = self.scalar<long long>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::f16: {
                auto d = self.scalar<af::half>();
                auto f = static_cast<float>(half::from_bits(d.data_));
                return std::complex<double>(static_cast<double>(f), 0.0);
            }
            case af::dtype::f32: {
                auto d = self.scalar<float>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::f64: {
                auto d = self.scalar<double>();
                return std::complex<double>(d, 0.0);
            }
            case af::dtype::u16: {
                auto d = self.scalar<unsigned short>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::u32: {
                auto d = self.scalar<unsigned int>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::u64: {
                auto d = self.scalar<unsigned long long>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            case af::dtype::u8: {
                auto d = self.scalar<unsigned char>();
                return std::complex<double>(static_cast<double>(d), 0.0);
            }
            default:
                throw std::runtime_error("Unexpected type");
                break;
            }
        });

    // ka.def("__iter__",
    //     [](const af::array &self){
    //         // Need to do this method for full compat with ndarray
    //         // and pandas.
    //         throw std::runtime_error("TODO");
    //     });

    ka.def("__repr__",
           [](const af::array &self) {
               self.eval();
               char *out = nullptr;
               af_array_to_string(&out, "", self.get(), 4, true);
               return std::string(out);
           });

    ka.def("display",
           [](const af::array &self, int precision) {
               self.eval();
               char *out = nullptr;
               af_array_to_string(&out, "", self.get(), precision, true);
               py::print(py::str(out));
           },
           py::arg("precision") = 4);

    ka.def("__matmul__",
           [](const af::array &self, const py::object &other) {
               af::array rhs = pygauss::arraylike::as_array_checked(other);

               af_array out = nullptr;
               throw_on_error(af_matmul(&out, self.get(), rhs.get(), AF_MAT_NONE, AF_MAT_NONE));
               return af::array(out);
           },
           py::arg("other").none(false),
           "Matrix multiplication");

    ka.def("__rmatmul__",
           [](const af::array &self, const py::object &other) {
               af::array rhs = pygauss::arraylike::as_array_checked(other);

               af_array out = nullptr;
               throw_on_error(af_matmul(&out, rhs.get(), self.get(), AF_MAT_NONE, AF_MAT_NONE));
               return af::array(out);
           },
           py::arg("other").none(false),
           "Matrix multiplication");


    ka.def("__abs__",
           [](const af::array &self) {
               return af::abs(self);
           });

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
           [](af::array &self) {
               auto zero = py::float_(0.0f);
               return binary_function(self, zero, true, af_sub);
           });

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
                  throw_on_error(af_eval_multiple(lst.size(), lst.data()));
              }
          },
          "Forces the evaluation of all the arrays");
}
