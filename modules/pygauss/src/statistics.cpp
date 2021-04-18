#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <pygauss.h>

namespace py = pybind11;
namespace gs = gauss::statistics;

void pygauss::bindings::statistic_functions(py::module &m)
{
    py::enum_<gs::XCorrScale>(m, "XCorrScale", "Cross Correlation Scale")
            .value("NoScale", gs::XCorrScale::NONE, "")
            .value("Biased", gs::XCorrScale::BIASED, "")
            .value("Unbiased", gs::XCorrScale::UNBIASED, "")
            .value("Coeff", gs::XCorrScale::COEFF, "")
            .export_values();

    m.def(
        "mean",
        [](const py::object &a, const py::object &weights, const std::optional<long long> dim) {
            auto array = arraylike::as_array_checked(a);
            auto hasWeights = !weights.is_none();
            auto isAggregated = !dim.has_value();
            
            std::variant<double, std::complex<double>, af::array> result;

            if (isAggregated)
            {
                double real, imag;
                af_err err;
                if (hasWeights) {
                    auto arrw = arraylike::as_array_checked(weights);
                    err = af_mean_all_weighted(&real, &imag, array.get(), arrw.get());
                }
                else
                    err = af_mean_all(&real, &imag, array.get());

                throw_on_error(err);

                if (array.iscomplex())
                    result = std::complex<double>(real, imag);
                else
                    result = real;
            }
            else
            {
                if (hasWeights)
                    result = af::mean(array, arraylike::as_array_checked(weights), dim.value());
                else
                    result = af::mean(array, dim.value());
            }

            return result;
        },
        py::arg("a").none(false),
        py::arg("weights") = py::none(),
        py::arg("dim") = py::none());

    m.def(
        "median",
        [](const py::object &a, const std::optional<long long> dim) {
            auto array = arraylike::as_array_checked(a);
            auto isAggregated = !dim.has_value();

            std::variant<double, std::complex<double>, af::array> result;
            if (isAggregated)
            {
                double real, imag;
                throw_on_error(af_median_all(&real, &imag, array.get()));
                if (array.iscomplex())
                    result = std::complex<double>(real, imag);
                else
                    result = real;
            }
            else
            {
                result = af::median(array, dim.value());
            }

            return result;
        },
        py::arg("a").none(false),
        py::arg("dim") = py::none());

    m.def(
        "var",
        [](const py::object &a, const py::object &weights, const std::optional<long long> dim, const bool biased) {
            auto array = arraylike::as_array_checked(a);
            auto hasWeights = !weights.is_none();
            auto isAggregated = !dim.has_value();

            std::variant<double, std::complex<double>, af::array> result;

            if (isAggregated)
            {
                double real, imag;
                af_err err;
                if (hasWeights)
                    err = af_var_all_weighted(&real, &imag, array.get(), arraylike::as_array_checked(weights).get());
                else
                    err = af_var_all(&real, &imag, array.get(), biased);

                throw_on_error(err);

                if (array.iscomplex())
                    result = std::complex<double>(real, imag);
                else
                    result = real;
            }
            else
            {
                if (hasWeights)
                    result = af::var(array, arraylike::as_array_checked(weights), dim.value());
                else
                    result = af::var(array, biased, dim.value());
            }

            return result;
        },
        py::arg("a").none(false),
        py::arg("weights") = py::none(),
        py::arg("dim") = py::none(),
        py::arg("biased") = false);

    m.def(
        "std",
        [](const py::object &a, const std::optional<long long> dim) {
            auto array = arraylike::as_array_checked(a);
            std::variant<double, std::complex<double>, af::array> result;
            auto isAggregated = !dim.has_value();

            if (isAggregated)
            {
                double real, imag;
                throw_on_error(af_stdev_all(&real, &imag, array.get()));

                if (array.iscomplex())
                    result = std::complex<double>(real, imag);
                else
                    result = real;
            }
            else
            {
                result = af::stdev(array, dim.value());
            }

            return result;
        },
        py::arg("a").none(false),
        py::arg("dim") = py::none());

    m.def(
        "skewness",
        [](const py::object &data) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);
            return gs::skewness(tss);
        },
        py::arg("data").none(false));

    m.def(
        "kurtosis",
        [](const py::object &data) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);
            return gs::kurtosis(tss);
        },
        py::arg("data").none(false));

    m.def(
        "moment",
        [](const py::object &data, const unsigned int k) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);
            return gs::moment(tss, k);
        },
        py::arg("data").none(false),
        py::arg("k").none(false));

    m.def(
        "cov",
        [](const py::object &data, const unsigned int ddof) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);
            return gs::covariance(tss, ddof);
        },
        py::arg("data").none(false),
        py::arg("ddof") = 1);

    m.def(
        "corrcoef",
        [](const py::object &data, const unsigned int ddof) {
            auto tss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(tss);
            return gs::corrcoef(tss, ddof);
        },
        py::arg("data").none(false),
        py::arg("ddof") = 1);

    m.def(
        "xcov",
        [](const py::object &xss, const py::object &yss, 
           const std::optional<unsigned int> &maxlag, 
           const std::optional<gs::XCorrScale> &scale) {

            auto x = arraylike::as_array_checked(xss);
            auto y = arraylike::as_array_checked(yss);
            arraylike::ensure_floating(x);
            arraylike::ensure_floating(y);
            return gs::xcov(x, y, maxlag, scale);
        },
        py::arg("xss").none(false),
        py::arg("yss").none(false),
        py::arg("maxlag") = py::none(),
        py::arg("scale") = gs::XCorrScale::NONE);

    m.def(
        "xcorr",
        [](const py::object &xss, const py::object &yss, 
           const std::optional<unsigned int> &maxlag, 
           const std::optional<gs::XCorrScale> &scale) {

            auto x = arraylike::as_array_checked(xss);
            auto y = arraylike::as_array_checked(yss);
            arraylike::ensure_floating(x);
            arraylike::ensure_floating(y);
            
            return gs::xcorr(x, y, maxlag, scale);
        },
        py::arg("xss").none(false),
        py::arg("yss").none(false),
        py::arg("maxlag") = py::none(),
        py::arg("scale") = gs::XCorrScale::NONE);

    m.def(
        "acorr",
        [](const py::object &data,  
           const std::optional<unsigned int> &maxlag, 
           const std::optional<gs::XCorrScale> &scale) {

            auto ss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(ss);
            return gs::autocorr(ss, maxlag, scale);
        },
        py::arg("data").none(false),
        py::arg("maxlag") = py::none(),
        py::arg("scale") = gs::XCorrScale::NONE);

    m.def(
        "acov",
        [](const py::object &data,  
           const std::optional<unsigned int> &maxlag, 
           const std::optional<gs::XCorrScale> &scale) {

            auto ss = arraylike::as_array_checked(data);
            arraylike::ensure_floating(ss);
            return gs::autocov(ss, maxlag, scale);
        },
        py::arg("data").none(false),
        py::arg("maxlag") = py::none(),
        py::arg("scale") = gs::XCorrScale::NONE);


//////

    // m.def(
    //     "ljungbox",
    //     [](const py::object &data, const unsigned int lags) {
    //         auto ss = arraylike::as_array_checked(data);
    //         arraylike::ensure_floating(ss);
    //         return gs::ljungBox(ss, lags);
    //     },
    //     py::arg("data").none(false),
    //     py::arg("lags").none(false));

    // m.def(
    //     "quantile",
    //     [](const py::object &data, const py::object& quantiles, const bool is_sorted){
    //         auto ss = arraylike::as_array_checked(data);
    //         arraylike::ensure_floating(ss);
    //         if (!is_sorted)
    //             ss = af::sort(ss, 0);

    //         auto qs = arraylike::as_itself_or_promote(quantiles, af::dim4(1), ss.type());
    //         return gs::quantile(ss, qs);
    //     },
    //     py::arg("data").none(false),
    //     py::arg("quantiles").none(false),
    //     py::arg("is_sorted") = false);

    // m.def(
    //     "quantiles_cut",
    //     [](const py::object &data, const unsigned int regions, const bool is_sorted) {
    //         auto ss = arraylike::as_array_checked(data);
    //         arraylike::ensure_floating(ss);
    //         if (!is_sorted)
    //             ss = af::sort(ss, 0);
    //         return gs::quantilesCut(ss, static_cast<float>(regions));
    //     },
    //     py::arg("data").none(false),
    //     py::arg("regions").none(false),
    //     py::arg("is_sorted") = false);

    m.def(
        "topk_max",
        [](const py::object &data, const int k) {
            auto arr = arraylike::as_array_checked(data);
            arraylike::ensure_floating(arr);
            af::array values;
            af::array indices;
            af::topk(values, indices, arr, k, 0, af_topk_function::AF_TOPK_MAX);
            
            py::tuple result(2);
            result[0] = values;
            result[1] = indices;
            return result;
        },
        py::arg("a").none(false),
        py::arg("k").none(false));

    m.def(
        "topk_min",
        [](const py::object &data, const int k) {
            auto arr = arraylike::as_array_checked(data);
            arraylike::ensure_floating(arr);
            af::array values;
            af::array indices;
            af::topk(values, indices, arr, k, 0, af_topk_function::AF_TOPK_MIN);

            py::tuple result(2);
            result[0] = values;
            result[1] = indices;
            return result;
        },
        py::arg("a").none(false),
        py::arg("k").none(false));
}
