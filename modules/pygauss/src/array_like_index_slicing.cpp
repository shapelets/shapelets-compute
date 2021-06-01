/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pygauss.h>

namespace py = pybind11;
namespace spd = spdlog;

/*

[9 1 1 1]
Negative Indices			   -9  -8  -7  -6  -5  -4  -3  -2  -1
Positive Indices    		    0   1   2   3   4   5   6   7   8

Values in vector			   10  20  30  40  50  60  70  80  90

numpy			arrayfire	   result
-------------	-----------    -----------------------------------
::-1			-1,  0, -1	   90  80  70  60  50  40  30  20  10
-1::-1			-1,  0, -1	   90  80  70  60  50  40  30  20  10
-1:-10:-1		-1, -9, -1	   90  80  70  60  50  40  30  20  10
-1:-9:-1		-1, -8, -1	   90  80  70  60  50  40  30  20
-2:-9:-1		-2, -8, -1	   80  70  60  50  40  30  20
-8:-9:-1		-8, -8, -1	   20
-1:-2:-1        -1, -1, -1     90
-9:-10:-1       -9, -9, -1     10
1:-1			 1, -2,  1     20  30  40  50  60  70  80
::				 0, -1,  1     10  20  30  40  50  60  70  80  90
1:				 1, -1,  1	   20  30  40  50  60  70  80  90
1:5				 1,  4,  1	   20  30  40  50

 */

namespace pygauss {

    std::tuple<dim_t, af::dim4, af_index_t *>
    build_index_internal(const py::object &selector, const af::dim4 &arr_dim) {

        if (selector.is_none())
            throw std::runtime_error("No selector");

        dim_t result_dimensions = 0;
        af_index_t *afIndex = nullptr;
        throw_on_error(af_create_indexers(&afIndex));

        auto result_dim = af::dim4(1);

        if (py::isinstance<py::ellipsis>(selector)) {
            // the user has done something like array[...]
            // which should result in a select all operation.
            spd::trace("Selector was just `...`; returning without further indexing");
            result_dim = arr_dim;
            result_dimensions = arr_dim.ndims();
            return {result_dimensions, result_dim, afIndex};
        }

        // Homogenize the input knowing that at max there are 4 dimensions
        // and remove single item cases and / or ellipsis in tuples
        std::vector<std::optional<py::object>> items(4, std::nullopt);

        if (!py::isinstance<py::tuple>(selector)) {
            // we are dealing with a single entry,
            // which is not an ellipsis
            items[0] = selector;
            result_dimensions = 1;
            spd::trace("Selector is a tuple of only one element");
        } else {
            auto tuple = py::cast<py::tuple>(selector);
            auto tuple_size = tuple.size();

            if (tuple_size == 0)
                // ensure we detect arr[()]
                throw std::runtime_error("Empty tuple is not a valid indexer");

            if (tuple_size > 4)
                throw std::runtime_error("Up to four dimensions are supported.  If more dimensions are required, "
                                         "cast to numpy using either a memoryview object or np.array(obj)");

            spd::trace("Selector is a tuple of {} elements", tuple_size);

            size_t i = 0; // tuple item
            size_t d = 0; // current dimension
            while (i < tuple_size) {
                auto item = py::cast<py::object>(tuple[i]);
                if (py::isinstance<py::ellipsis>(item)) {
                    // if it is an ellipsis, calculate how many dimensions
                    // are we skipping based on the size of the tuple,
                    // the current position within the tuple and the
                    // total number of indexable dimensions
                    //
                    auto next_d = arr_dim.ndims() - tuple_size + i + 1;
                    // an ellipsis forces us to include the dimensions
                    spd::trace("Detected ellipsis in tuple at position {}...", i);
                    for (auto j = d; j < next_d; j++) {
                        spd::trace("\t...taking {} elements at dimension {}", arr_dim[j], j);
                        result_dim[j] = arr_dim[j];
                        result_dimensions += 1;
                    }
                    d = next_d;
                } else {
                    // store for further processing
                    items[d] = std::move(item);
                    d += 1;
                    result_dimensions += 1;
                }
                i += 1;
            }
        }

        // right now we have an object per dimension or empty option
        // it should be as simple as process each one of them:
        for (auto i = 0; i < 4; i++) {
            auto item_op = items[i];
            if (!item_op.has_value()) {
                spd::trace("Preprocessed selector: no value at position {}", i);
                continue;
            }
            auto item = item_op.value();

            if (item.is_none()) {
                spd::trace("Preprocessed selector: none value at position {}", i);
                continue;
            }

            if (py::isinstance<py::int_>(item)) {
                auto v = py::cast<int>(item);
                spd::trace("Preprocessed selector: found a integer at pos {} interpreted as {}, {}, {}", i, v, v, 1);
                throw_on_error(af_set_seq_param_indexer(afIndex, v, v, 1, i, false));
                spd::trace("Preprocessed selector: resulting dimension at pos {} is 1", i);
                result_dim[i] = 1;
                continue;
            }

            if (py::isinstance<py::slice>(item)) {
                auto slice = Slice::from_python(py::cast<py::slice>(item));
                auto start = slice.start();
                auto stop = slice.stop();
                auto step = slice.step();

                spd::trace("Preprocessed selector: found a slice at pos {} interpreted as {}, {}, {}", i, start, stop,
                           step);
                throw_on_error(af_set_seq_param_indexer(afIndex, start, stop, step, i, false));
                auto v1 = start < 0 ? arr_dim[i] + start : start;
                auto v2 = stop < 0 ? arr_dim[i] + stop : stop;
                auto span = static_cast<int32_t>(std::abs(v1 - v2)) + 1;
                auto int_div = std::div(span, static_cast<int32_t>(std::abs(step)));
                auto len = int_div.quot + (int_div.rem != 0 ? 1 : 0);
                result_dim[i] = (dim_t) len;
                spd::trace("Preprocessed selector: resulting dimension at pos {} is {} (Original was {})", i,
                           result_dim[i], arr_dim[i]);
                continue;
            }

            if (py::isinstance<ParallelFor>(item)) {
                auto pf = py::cast<ParallelFor>(item).slice();
                auto start = pf.start();
                auto stop = pf.stop();
                auto step = pf.step();
                spd::trace("Preprocessed selector: found a parallel for at pos {} interpreted as {}, {}, {}", i, start,
                           stop, step);
                throw_on_error(af_set_seq_param_indexer(afIndex, start, stop, step, i, true));
                auto v1 = start < 0 ? arr_dim[i] + start : start;
                auto v2 = stop < 0 ? arr_dim[i] + stop : stop;
                auto span = static_cast<int32_t>(std::abs(v1 - v2) + 1);
                auto int_div = std::div(span, static_cast<int32_t>(std::abs(step)));
                auto len = int_div.quot + (int_div.rem != 0 ? 1 : 0);
                result_dim[i] = (dim_t) len;
                spd::trace("Preprocessed selector: resulting dimension at pos {} is {} (Original was {})", i,
                           result_dim[i], arr_dim[i]);
                continue;
            }

            if (py::isinstance<af::array>(item)) {
                auto ind_arr = py::cast<af::array>(item);
                af_array out;
                if (ind_arr.type() == af::dtype::b8) {
                    throw_on_error(af_where(&out, ind_arr.get()));
                    double re,im;
                    throw_on_error(af_sum_all(&re, &im, ind_arr.get()));
                    result_dim[i] = static_cast<dim_t>(re);
                    spd::trace("Preprocessed selector: found boolean array at pos {}", i);
                } else {
                    throw_on_error(af_retain_array(&out, ind_arr.get()));
                    throw_on_error(af_get_elements(&result_dim[i], out));
                    spd::trace("Preprocessed selector: found any array at pos {}", i);
                }

                throw_on_error(af_set_array_indexer(afIndex, out, i));
                spd::trace("Preprocessed selector: resulting dimension at pos {} is {} (Original was {})", i,
                           result_dim[i], arr_dim[i]);
                continue;
            }

            throw std::runtime_error("Unable to process index expression");
        }

        spd::trace("Indexing calculations ended. Result dimensions ({}) {}", result_dimensions, result_dim);
        return {result_dimensions, result_dim, afIndex};
    }


    std::tuple<dim_t, af::dim4, af_index_t *>
    arraylike::build_index(const py::object &selector, const af::dim4 &arr_dim) {
        auto[res_dim, index_dim, index] = build_index_internal(selector, arr_dim);

        if (spd::get_level() == spd::level::debug) {
            for (auto i = 0; i < 4; i++) {
                auto index_entry = index[i];
                if (index_entry.isSeq) {
                    spd::debug("\t{}\t{}: Sequence {}-{}-{} with dimensions {}",
                               i < res_dim ? "In" : "Out",
                               i,
                               index_entry.idx.seq.begin,
                               index_entry.idx.seq.end, index_entry.idx.seq.step,
                               af::calcDim(index_entry.idx.seq, arr_dim[i]));
                } else {
                    spd::debug("\t\t{}: Array", i);
                }
            }
        }

        return {res_dim, index_dim, index};
    }

}
