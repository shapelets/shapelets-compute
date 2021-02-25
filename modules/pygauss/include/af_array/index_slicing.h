#ifndef AF_ARRAY_INDEX__H
#define AF_ARRAY_INDEX__H

#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace spd = spdlog;

class GForStatus {
private:
    GForStatus() = default;

    thread_local static bool status;
public:
    static bool get() {
        spd::debug("Accessing GForStatus.  Value is {}", status);
        return status;
    }
    static void set(bool newValue) {
        spd::debug("Setting GForStatus.  New Value is {}; old value is {}", newValue, status);
        status = newValue;
    }
    static void toggle() {
        status ^= true;
        spd::debug("Toggle GForStatus.  New Value is {}", status);
    }
};

class gfor_adquire {
public:
    gfor_adquire() {
        GForStatus::set(true);
    }
    ~gfor_adquire() {
        GForStatus::set(false);
    }
};


class ParallelFor {
public:
    explicit ParallelFor(ssize_t value): start(0), stop(value-1), step(1)  {
        // empty
    }

    explicit ParallelFor(const py::slice& slice);

    ParallelFor(const ParallelFor& src) {
        start = src.start;
        stop = src.stop;
        step = src.step;
    }

    ssize_t getStart() const { return start; }
    ssize_t getStop() const  { return stop; }
    ssize_t getStep() const { return step;}

private:
    ssize_t start;
    ssize_t stop;
    ssize_t step;
};

typedef struct result {
    ssize_t start;
    ssize_t stop;
    ssize_t step;
} slice_interpretation;

slice_interpretation interpret_slice(const py::slice &slice);

//af::dim4 assign_dimensions(const py::object& selector, const af::dim4& original_dimensions);
//af_index_t* build_index(const py::object& selector, int array_num_dimensions);

std::tuple<af::dim4,af_index_t*> build_index(const py::object& selector, const af::dim4& arr_dim);

#endif // AF_ARRAY_INDEX__H

