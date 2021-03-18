#ifndef __PYGAUSS_H__
#define __PYGAUSS_H__

#include <half.hpp>
#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <gauss.h>

#include "formatters.h"
#include "typecasters.h"
#include "templates.h"

namespace py = pybind11;
namespace spd = spdlog;

namespace pygauss {

    /**
     * Controls if arrayfire gfor or broadcast flag is on to deal correctly with
     * certain API calls and constructs.
     */
    class GForStatus {
    private:
        /**
         * It is all static access...
         */
        GForStatus() = default;

        /**
         * Stores the status as thread local
         */
        thread_local inline static auto status = false;

    public:
        static bool get() {
            spd::trace("Accessing GForStatus.  Value is {}", status);
            return status;
        }

        static void set(bool newValue) {
            spd::trace("Setting GForStatus.  New Value is {}; old value is {}", newValue, status);
            status = newValue;
        }

        static void toggle() {
            status ^= true;
            spd::trace("Toggle GForStatus.  New Value is {}", status);
        }
    };

    /**
     * Pattern for scoped resource acquisition to control the value of GForStatus
     */
    class gfor_acquire {
    private:
        /**
         * Remembers if GForStatus was already set when entering this
         * scope; if so, no changes are performed so composition can
         * be done.
         */
        bool already_enabled;
    public:
        gfor_acquire() {
            already_enabled = GForStatus::get();
            if (!already_enabled)
                GForStatus::set(true);
        }

        /**
         * Leaves the scope
         */
        ~gfor_acquire() {
            if (already_enabled)
                GForStatus::set(false);
        }
    };

    class Slice {
    public:

        explicit Slice(ssize_t start, ssize_t stop, ssize_t step) :
                _start(start), _stop(stop), _step(step) {}

        Slice(const Slice &src) :
                _start(src.start()), _stop(src.stop()), _step(src.step()) {}

        /**
        * Gets the starting point of the iteration.
        */
        ssize_t start() const { return _start; }

        /**
         * Gets the ending point of the iteration (included)
         */
        ssize_t stop() const { return _stop; }

        /**
         * Gets the step of the iteration
         */
        ssize_t step() const { return _step; }

        /**
         * Parses a py::slice object and returns a slice compatible with arrayfire
         * @param slice
         * @return
         */
        static Slice from_python(const py::slice &slice) {

            if (slice.is_none())
                throw std::runtime_error("Null value given for slice");

            // default values
            ssize_t _start = 0;
            ssize_t _end = -1;
            ssize_t _step = 1;

            auto raw = (PySliceObject *) slice.ptr();
            auto raw_step = py::reinterpret_borrow<py::int_>(raw->step);
            auto raw_start = py::reinterpret_borrow<py::int_>(raw->start);
            auto raw_stop = py::reinterpret_borrow<py::int_>(raw->stop);

            if (!raw_step.is_none()) {
                _step = raw_step.cast<ssize_t>();
                if (_step < 0) {
                    _start = -1;
                    _end = 0;
                }
            }

            if (!raw_start.is_none())
                _start = raw_start.cast<ssize_t>();

            if (!raw_stop.is_none())
                _end = raw_stop.cast<ssize_t>();

            if (_start >= 0 && _end >= 0 && _end <= _start && _step >= 0) {
                _start = 1;
                _end = 1;
                _step = 1;
            } else if (_start < 0 && _end < 0 && _end >= _start && _step <= 0) {
                _start = -2;
                _end = -2;
                _step = -1;
            }

            if (!raw_stop.is_none())
                _end = _end + (_end < 0 ? 1 : -1);

            return Slice(_start, _end, _step);
        }

    private:
        ssize_t _start;
        ssize_t _stop;
        ssize_t _step;
    };

    /**
     * Class representing a parallel for construct, which could be converted
     * to an af::array allowing index semantics to participate in loops.
     *
     * Class is immutable.
     */
    class ParallelFor {
    public:

        /**
         * Constructor from a single value representing the total number of
         * parallel / concurrent iterations to execute.
         */
        explicit ParallelFor(ssize_t value) :
                _slice(Slice(0, value - 1, 1)) {}

        /**
         * Constructor from an explicit iteration sequence / slice.
         */
        explicit ParallelFor(ssize_t start, ssize_t stop, ssize_t step) :
                _slice(Slice(start, stop, step)) {}

        explicit ParallelFor(const Slice &slice) :
                _slice(slice) {}

        /**
         * Copy constructor
         */
        ParallelFor(const ParallelFor &src) :
                _slice(src.slice()) {}

        /**
         * Access slice information
         */
        [[nodiscard]] const Slice &slice() const { return _slice; }

        /**
         * Converts the construction to an array that can participate in
         * the iterations parallel iterations.
         */
        explicit operator af::array() const {
            return af::seq(_slice.start(), _slice.stop(), _slice.step());
        }

    private:
        Slice _slice;
    };

    namespace arraylike {

        /**
         * Returns the size in bytes of an element of type `dtype`
         */
        int scalar_size(const af::dtype &dtype);

        /**
         * Analysis of a python indexer and conversion to a af_index_t structure.
         *
         * It also calculates the resulting dimensionality of the selection and returns
         * all that information in a single tuple, where:
         *  - 0: is the total number of dimensions of the index
         *  - 1: is the actual dimensionality (axes by axes)
         *  - 2: A pointer to an array fire af_index_t of 4 elements (one per dimension)
         *       that can be utilized in __getitem__ and __setitem__ operations.
         */
        std::tuple<dim_t, af::dim4, af_index_t *> build_index(const py::object &selector, const af::dim4 &arr_dim);

        /**
         * Exposes an af array through the buffer protocol
         */
        py::buffer_info numpy_buffer_protocol(const af::array &self);

        /**
         * Returns true if the object is a scalar, that is, if the
         * object represents either an py::int_, py::float_,
         * py::bool_ or std::complex<double>
         */
        bool is_scalar(const py::object &value);

        /**
         * Checks if src is of type floating (that is any floating or complex type).  If
         * it is not, it converts src to a new array whose type is either f32 or f64 as a function
         * of the currently selected device.
         */
        void ensure_floating(af::array& src, bool warn_if_conversion = true);

        std::optional<af::array> as_array(const py::object& obj);

        af::array as_itself_or_promote(const py::object& obj, const af::dim4 &shape, const af::dtype& ref_type);

        af::array as_array_like(const py::object& obj, const std::optional<af::dim4> &shape,
                                const std::optional<af::dtype> &dtype);

        af::array as_array_like(const py::object& obj, const af::array& like);

        af::array as_array_checked(const py::object& obj);

        std::optional<af::array> scalar_as_array(const py::object& obj, const af::dim4& shape, const af::dtype& ref_type);
        af::array scalar_as_array_checked(const py::object& obj, const af::dim4& shape, const af::dtype& ref_type);

        std::optional<std::pair<af::array, af::array>> as_array(const py::object &x, const py::object &y);

        /**
         * The inner detail namespace is where specializations happen
         */
        namespace detail {

            af::dtype harmonize_types(const af::dtype scalar_type, const af::dtype array_type);

            bool numpy_is_scalar(const py::object& value);
            af::array numpy_scalar_to_array(const py::object& value, const af::dim4 &shape, const af::dtype& ref_type);

            bool python_is_scalar(const py::object& value);
            af::array python_scalar_to_array(const py::object& value, const af::dim4 &shape, const af::dtype& ref_type);

            bool numpy_is_array(const py::object& value);
            af::array numpy_array_to_array(const py::object &value);

            bool af_is_array(const py::object &value);
            af::array from_af_array(const py::object &value);
        }
    }

    namespace bindings {
        void device_operations(py::module &m);

        void shared_enum_types(py::module &m);

        void array_obj(py::module &m);

        void array_construction_operations(py::module &m);

        void extract_and_transform_operations(py::module &m);

        void batch_api(py::module &m);

        void parallel_algorithms(py::module &m);

        void linear_algebra_operations(py::module &m);

        void logic_operations(py::module &m);

        void math_operations(py::module &m);

        void random_numbers(py::module &m);

        void signal_processing_functions(py::module &m);

        void statistic_functions(py::module &m);

        void matrix_profile_functions(py::module &m);

        void gauss_distance_functions(py::module &m);

        void gauss_statistic_bindings(py::module &m);

        void gauss_normalization_functions(py::module &m);
    }


    void warn_if_error(const af_err &err);

    void throw_on_error(const af_err &err);
}

#endif  // __PYGAUSS_H__
