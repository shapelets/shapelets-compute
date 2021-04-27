#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <variant>
#include <pygauss.h>

namespace py = pybind11;

af::array linspace(af::array &s, af::array &e, int num, bool endpoint, int axis) {
    if (axis >= 4)
        throw std::invalid_argument("Axis should be in range [0..3]");

    if (e.type() != s.type()) {
        if (s.iscomplex() && !e.iscomplex())
            e = e.as(s.type());
        else if (e.iscomplex() && !s.iscomplex()) 
            s = s.as(e.type());
        else if (s.isfloating() && !e.isfloating()) 
            e = e.as(s.type());
        else 
            s = s.as(e.type());
    }

    if (s.numdims() == 1 && e.numdims() > 1) {
        s = af::tile(s, e.dims());
    } else if (e.numdims() == 1 && s.numdims() > 1) {
        e = af::tile(e, s.dims());
    } else if (e.dims() != s.dims()) {
        throw std::invalid_argument("Incompatible dimensions between start and end");
    }

    if (num == 1)
        return s;
    
    auto delta = e - s;
    auto inc = delta / ((endpoint) ? num - 1 : num);

    auto newDims = af::dim4(s.dims());
    if (axis < newDims.ndims()) {
        for(auto i = 2; i>=axis; i--) {
            newDims[i+1] = newDims[i];
        }
        newDims[axis] = 1;
    }

    // af::array arr_objs[num];
    af::array *arr_objs = alloca(num * sizeof(af::array));
    arr_objs[0] = af::moddims(s.copy(), newDims);
    for (auto i=1; i < num; i++) {
        s = s + inc;
        arr_objs[i] = af::moddims(s, newDims);
    }

    af::array acc = arr_objs[0];
    auto i = 1;
    while (i < num) {
        auto left = num - i;
        if (left >= 3) {
            acc = af::join(axis, acc, arr_objs[i], arr_objs[i+1], arr_objs[i+2]);
            i += 3;
        } else if (left >= 2) {
            acc = af::join(axis, acc, arr_objs[i], arr_objs[i+1]);
            i += 2;
        } else {
            acc = af::join(axis, acc, arr_objs[i]);
            i += 1;
        }
    }
    return acc;
};


void pygauss::bindings::array_construction_operations(py::module &m)
{
    m.def(
        "geomspace",
        [](const py::object &start, const py::object &end, int num, bool endpoint, int axis, const af::dtype &dtype) {
            auto comp_type = af::isDoubleAvailable(af::getDevice()) ? af::dtype::f64 : af::dtype::f32;
            auto s = arraylike::as_itself_or_promote(start, af::dim4(1), comp_type);
            auto e = arraylike::as_itself_or_promote(end, af::dim4(1), comp_type);
            
            if (af::anyTrue<bool>(af::iszero(s) || af::iszero(e)))
                throw std::invalid_argument("Zero values are not supported in geometric space");

            auto log_s = af::log10(s);
            auto log_e = af::log10(e);
            auto r = af::pow(10.0, linspace(log_s, log_e, num, endpoint, axis));
            if (r.type() != dtype) 
                return r.as(dtype);
            return r;                
        },
        py::arg("start").none(false),
        py::arg("end").none(false),
        py::arg("num") = 50,
        py::arg("endpoint") = true,
        py::arg("axis") = 0,
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "logspace",
        [](const py::object &start, const py::object &end, int num, bool endpoint, int axis, double base, const af::dtype &dtype ) {
            auto s = arraylike::as_itself_or_promote(start, af::dim4(1), dtype);
            auto e = arraylike::as_itself_or_promote(end, af::dim4(1), dtype);
            return af::pow(base, linspace(s, e, num, endpoint, axis));
        },
        py::arg("start").none(false),
        py::arg("end").none(false),
        py::arg("num") = 50,
        py::arg("endpoint") = true,
        py::arg("axis") = 0,
        py::arg("base") = 10.0,
        py::arg("dtype") = af::dtype::f32           
    );

    m.def(
        "linspace", 
        [](const py::object &start, const py::object &end, int num, bool endpoint, int axis, const af::dtype &dtype) {
            auto s = arraylike::as_itself_or_promote(start, af::dim4(1), dtype);
            auto e = arraylike::as_itself_or_promote(end, af::dim4(1), dtype);
            return linspace(s, e, num, endpoint, axis);

        },
        py::arg("start").none(false),
        py::arg("end").none(false),
        py::arg("num") = 50,
        py::arg("endpoint") = true,
        py::arg("axis") = 0,
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "arange",
        [](const std::variant<py::int_, py::float_> &stop, const af::dtype &dtype) {
            auto cstop = stop.index() == 0 
                ? py::cast<double>(std::get<py::int_>(stop)) 
                : py::cast<double>(std::get<py::float_>(stop));

            auto len = static_cast<long long>(std::ceil(cstop));
            return af::iota(len, af::dim4(1), dtype);
        },
        py::arg("stop").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "arange",
        [](const std::variant<py::int_, py::float_> &start, 
           const std::variant<py::int_, py::float_> &stop, 
           const std::variant<py::int_, py::float_> &step, 
           const af::dtype &dtype) {

            auto cstart = start.index() == 0 
                ? py::cast<double>(std::get<py::int_>(start)) 
                : py::cast<double>(std::get<py::float_>(start));

            auto cstop = stop.index() == 0 
                ? py::cast<double>(std::get<py::int_>(stop)) 
                : py::cast<double>(std::get<py::float_>(stop));

            auto cstep = step.index() == 0 
                ? py::cast<double>(std::get<py::int_>(step)) 
                : py::cast<double>(std::get<py::float_>(step));

            auto len = static_cast<long long>(std::ceil((cstop - cstart) / cstep));
            return cstart + (af::iota(len, af::dim4(1), dtype) * cstep);
        },
        py::arg("start").none(false),
        py::arg("stop").none(false),
        py::arg("step").none(false),
        py::arg("dtype") = af::dtype::f32);


    m.def(
        "empty",
        [](const af::dim4 &shape, const af::dtype &dtype) {
            return af::array(shape, dtype);
        },
        py::arg("shape").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "eye",
        [](const int N, const std::optional<int> &M, const int k, const af::dtype &dtype) {
            if (N <= 0)
                throw std::invalid_argument("N cannot be negative or zero");

            auto n = N;
            auto m = M.value_or(N);

            auto one_count = std::min(n, m) - std::abs(k);
            auto ones = af::constant(1, one_count, dtype);
            auto result = af::diag(ones, k, false);
            auto result_rows = result.dims(0);
            auto result_cols = result.dims(1);
            if (result_rows < n || result_cols < m)
            {
                auto end_padding = af::dim4(n - result_rows, m - result_cols, 0, 0);
                result = af::pad(result, af::dim4(0, 0, 0, 0), end_padding, af::borderType::AF_PAD_ZERO);
            }
            return result;
        },
        py::arg("N").none(false),
        py::arg("M") = py::none(),
        py::arg("k") = 0,
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "identity",
        [](const af::dim4 &shape, const af::dtype &dtype) {
            return af::identity(shape, dtype);
        },
        py::arg("shape").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "full",
        [](const af::dim4 &shape, const py::object &fill_value, const af::dtype &dtype) {
            auto candidate = arraylike::scalar_as_array(fill_value, shape, dtype);
            if (!candidate)
                throw std::invalid_argument("Unable to create array");

            auto result = candidate.value();
            if (result.type() == dtype)
                return result;

            return result.as(dtype);
        },
        py::arg("shape").none(false),
        py::arg("fill_value").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "zeros",
        [](const af::dim4 &shape, const af::dtype &dtype) {
            return af::constant(0, shape, dtype);
        },
        py::arg("shape").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "ones",
        [](const af::dim4 &shape, const af::dtype &dtype) {
            return af::constant(1, shape, dtype);
        },
        py::arg("shape").none(false),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "diag",
        [](const py::object &arr_like, int index = 0, bool extract = false) {
            auto a = arraylike::as_array_checked(arr_like);
            return af::diag(a, index, extract);
        },
        py::arg("a").none(false),
        py::arg("index") = 0,
        py::arg("extract") = false);

    m.def(
        "array",
        [](const py::object &array_like,
           const std::optional<af::dim4> &shape,
           const std::optional<af::dtype> &dtype) {
            return arraylike::as_array_like(array_like, shape, dtype);
        },
        py::arg("array_like").none(false),
        py::arg("shape") = py::none(),
        py::arg("dtype") = py::none());

    m.def(
        "iota",
        [](const af::dim4 &shape, const af::dim4 &tile, const af::dtype &dtype) {
            return af::iota(shape, tile, dtype);
        },
        py::arg("shape").none(false),
        py::arg("tile") = af::dim4(1),
        py::arg("dtype") = af::dtype::f32);

    m.def(
        "range", [](const af::dim4 &shape, int seq_dim, const af::dtype &dtype) {
            return af::range(shape, seq_dim, dtype);
        },
        py::arg("shape").none(false), 
        py::arg("seq_dim") = -1, 
        py::arg("dtype") = af::dtype::f32);

}
