/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/features.h>
#include <gauss/normalization.h>
#include <gauss/polynomial.h>
#include <gauss/regression.h>
#include <gauss/regularization.h>
#include <gauss/statistics.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

using CWTTuple = std::tuple<std::vector<int>, std::vector<int>, int>;
using LineTuple = std::tuple<std::vector<int>, std::vector<int>>;

using namespace gauss::features;

constexpr dim_t BATCH_SIZE = 2048;

namespace {

af::array entropy(af::array tss, int m, float r) {
    long n = static_cast<long>(tss.dims(0));

    af::array std = af::stdev(tss);
    std *= r;

    const long chunkSizeH = 512;
    const long chunkSizeV = 512;
    long lastIterationSizeH = chunkSizeH;
    long lastIterationSizeV = chunkSizeV;

    af::array expandH = af::array(m, chunkSizeH, tss.dims(1), tss.type());
    af::array expandV = af::array(m, chunkSizeV, tss.dims(1), tss.type());

    // Calculate the matrix distance
    af::array distances = af::constant(af::Inf, chunkSizeV, chunkSizeH, tss.dims(1), tss.type());
    af::array sum_c = af::constant(0, 1, chunkSizeH, tss.dims(1), tss.type());
    af::array sum = af::constant(0, tss.dims(1), tss.type());

    // it performs a batching (or blocking) in the horizontal direction
    for (int i = 0; i < n - m + 1; i += chunkSizeH) {
        long startH = i;
        long iterationSizeH = std::min(chunkSizeH, n - m + 1 - startH);
        // if our batching dimension does not match the dimesion of the remaining elements, modify dimensions
        if (iterationSizeH != chunkSizeH || iterationSizeH != lastIterationSizeH) {
            lastIterationSizeH = iterationSizeH;
            distances = af::constant(af::Inf, chunkSizeV, iterationSizeH, tss.dims(1), tss.type());
            sum_c = af::constant(0, 1, iterationSizeH, tss.dims(1), tss.type());
            expandH = af::array(m, iterationSizeH, tss.dims(1), tss.type());
        }

        gfor(af::seq j, static_cast<double>(tss.dims(1))) {
            for (int k = 0; k < m; k++) {
                expandH(k, af::span, j, af::span) =
                    af::reorder(tss(af::seq(i + k, i + k + iterationSizeH - 1), j), 1, 0, 3, 2);
            }
        }
        // it performs a batching (or blocking) in the vertical direction
        for (int j = 0; j < n - m + 1; j += chunkSizeV) {
            long startV = j;
            long iterationSizeV = std::min(chunkSizeV, n - m + 1 - startV);
            // if our batching dimension does not match the dimension of the remaining elements, modify dimensions
            if (iterationSizeV != chunkSizeV || iterationSizeV != lastIterationSizeV) {
                lastIterationSizeV = iterationSizeV;
                distances = af::constant(af::Inf, iterationSizeV, chunkSizeH, tss.dims(1), tss.type());
                expandV = af::array(m, iterationSizeV, tss.dims(1), tss.type());
            }

            gfor(af::seq k, static_cast<double>(tss.dims(1))) {
                for (int l = 0; l < m; l++) {
                    expandV(l, af::span, k, af::span) =
                        af::reorder(tss(af::seq(j + l, j + l + iterationSizeV - 1), k), 1, 0, 3, 2);
                }
            }
            // Get the maximum difference among all dimensions for each timeseries
            gfor(af::seq k, iterationSizeV) {
                af::array aux = af::tile(expandV(af::span, k, af::span), 1, static_cast<unsigned int>(iterationSizeH));
                distances = af::reorder(
                    af::max(af::abs(aux - af::tile(expandH, 1, 1, 1, static_cast<unsigned int>(iterationSizeV)))), 3, 1,
                    2, 0);
            }
            // sum the number of elements bigger than the threshhold given by (stdev*r)
            af::array count =
                distances <= af::tile(af::reorder(std, 0, 2, 1, 3), static_cast<unsigned int>(distances.dims(0)),
                                      static_cast<unsigned int>(distances.dims(1)));
            // we summarise all partial sums in sum; we accumulate all partial sums for each vertical dimension
            sum_c += af::sum(count, 0);
        }
        // Finally, we collapse (aggregate) all total sums for each timeseries, we can do it as sum is commutative and
        // associative.
        sum += af::reorder(af::sum(af::log(sum_c / (n - m + 1.0)), 1), 2, 0, 1, 3);
    }
    // Before returning the final value, we divide the total sum for each timeseries by (n - m + 1.0)
    sum /= (n - m + 1.0);

    return sum;
}

// af::array aggregatingOnChunks(const af::array &input, long chunkSize, AggregationFuncInt aggregationFunction) {
//     // Calculating the chunk size to split the input data into. The rest of dividing the input data
//     // length by the chunk size should be zero. In other words, input data length should be multiple
//     // of the chunk size
//     auto size = (input.dims(0) % chunkSize == 0) ? 0 : chunkSize - input.dims(0) % chunkSize;
//     // If it is not multiple, then pad with zeros at the end
//     af::array inputChunks = af::join(0, input, af::constant(0, size, input.dims(1), input.type()));
//     // Modify the array dimensions to split it by the chunk size
//     inputChunks = af::moddims(inputChunks, chunkSize, inputChunks.dims(0) / chunkSize, inputChunks.dims(1));
//     // Aggregate the data using the aggregation function specified as parameter, and then reorder
//     // the resulting array to have the number of chunks in the 2nd dimension
//     return af::reorder(af::transpose(aggregationFunction(inputChunks, 0)), 0, 2, 1, 3);
// }

// af::array aggregatingOnChunks(const af::array &input, long chunkSize, AggregationFuncDimT aggregationFunction) {
//     // Calculating the chunk size to split the input data into. The rest of dividing the input data
//     // length by the chunk size should be zero. In other words, input data length should be multiple
//     // of the chunk size
//     auto size = (input.dims(0) % chunkSize == 0) ? 0 : chunkSize - input.dims(0) % chunkSize;
//     // If it is not multiple, then pad with zeros at the end
//     af::array inputChunks = af::join(0, input, af::constant(0, size, input.dims(1), input.type()));
//     // Modify the array dimensions to split it by the chunk size
//     inputChunks = af::moddims(inputChunks, chunkSize, inputChunks.dims(0) / chunkSize, inputChunks.dims(1));
//     // Aggregate the data using the aggregation function specified as parameter, and then reorder
//     // the resulting array to have the number of chunks in the 2nd dimension
//     return af::reorder(af::transpose(aggregationFunction(inputChunks, 0)), 0, 2, 1, 3);
// }

af::array ricker(int points, int a) {
    double pi = 3.14159265358979323846264338327950288;
    double A = 2 / (std::sqrt(3 * a) * std::pow(pi, 0.25));
    double wsq = std::pow(a, 2);
    af::array vec = af::range(af::dim4(points)) - ((points - 1) / 2.0);
    af::array xsq = af::pow(vec, 2);
    af::array mod = (1 - xsq / wsq);
    af::array gauss = af::exp(-xsq / (2 * wsq));
    af::array total = A * mod * gauss;
    return total;
}

af::array cwt(const af::array &data, af::array widths) {
    int nw = static_cast<int>(widths.dims(0));
    int len_data = static_cast<int>(data.dims(0));
    int cols = static_cast<int>(data.dims(1));
    af::array filter;
    af::array output = af::constant(0, widths.dims(0), data.dims(0), data.dims(1), af::dtype::f32);

    for (int i = 0; i < nw; i++) {
        int w = widths(i).scalar<int>();
        int minimum = std::min(10 * w, len_data);
        af::array wavelet_data = ricker(minimum, w);
        if (wavelet_data.dims(0) != data.dims(0)) {
            wavelet_data = af::join(0, af::constant(0, 1, wavelet_data.dims(1)), wavelet_data);
        }
        output(i, af::span, af::span) =
            af::moddims(af::convolve(af::reorder(data, 0, 2, 1), wavelet_data), 1, len_data, cols);
    }
    return output;
}

af::array calculateMoment(const af::array &tss, int moment) {
    af::array output;
    af::array a = af::tile(af::pow(af::range(tss.dims(0)), moment), 1, static_cast<unsigned int>(tss.dims(1)));
    return af::sum(tss * a, 0) / af::sum(tss, 0);
}

af::array calculateCentroid(const af::array &tss) { return calculateMoment(tss, 1); }

af::array calculateVariance(const af::array &tss) {
    return calculateMoment(tss, 2) - af::pow(calculateCentroid(tss), 2);
}

af::array calculateSkew(const af::array &tss) {
    // In the limit of a dirac delta, skew should be 0 and variance 0. However, in the discrete limit,
    // the skew blows up as variance-- > 0, hence return nan when variance is smaller than a resolution of 0.5:
    af::array variance = calculateVariance(tss);
    af::array cond = variance < 0.5;
    af::array b = af::constant(af::NaN, 1, tss.dims(1), tss.type());
    af::array a =
        (calculateMoment(tss, 3) - 3 * calculateCentroid(tss) * variance - af::pow(calculateCentroid(tss), 3)) /
        af::pow(calculateVariance(tss), 1.5).as(tss.type());
    af::array out = af::select(cond, b, a);
    return out;
}

af::array calculateKurtosis(const af::array &tss) {
    // In the limit of a dirac delta, kurtosis should be 3 and variance 0. However, in the discrete limit,
    // the kurtosis blows up as variance-- > 0, hence return nan when variance is smaller than a resolution of 0.5:
    af::array variance = calculateVariance(tss);
    af::array cond = variance < 0.5;
    af::array b = af::constant(af::NaN, 1, tss.dims(1), tss.type());
    af::array a = (calculateMoment(tss, 4) - 4 * calculateCentroid(tss) * calculateMoment(tss, 3) +
                   6 * calculateMoment(tss, 2) * af::pow(calculateCentroid(tss), 2) - 3 * calculateCentroid(tss)) /
                  af::pow(calculateVariance(tss), 2).as(tss.type());
    af::array out = af::select(cond, b, a);
    return out;
}

// af::array estimateFriedrichCoefficients(af::array tss, int m, float r) {
//     auto n = tss.dims(0);

//     af::array categories = gauss::statistics::quantilesCut(tss(af::seq(n - 1), af::span), r);

//     af::array x = af::join(1, categories, af::reorder(tss(af::seq(n - 1), af::span), 0, 2, 1, 3));
//     x = af::join(1, x, af::reorder(af::diff1(tss, 0), 0, 2, 1, 3));

//     // Doing the groupBy per input time series in tss (tss are contained along the 2nd dimension).
//     // The groupBy function cannot be applied to all time series in parallel because we do not
//     // know a priori the number of groups in each series.

//     af::array result = af::array(m + 1, tss.dims(1), tss.type());

//     for (int i = 0; i < tss.dims(1); i++) {
//         af::array groupped = gauss::regularization::groupBy(x(af::span, af::span, i), af::mean, 2, 2);
//         result(af::span, i) = gauss::polynomial::polyfit(groupped.col(0), groupped.col(1), m);
//     }

//     return result;
// }

int indexMinValue(std::vector<int> values) {
    int result = -1;
    int minimum = std::numeric_limits<int>::max();
    for (size_t i = 0; i < values.size(); i++) {
        if (values[i] < minimum) {
            minimum = values[i];
            result = static_cast<int>(i);
        }
    }
    return result;
}

std::vector<int> subsValueToVector(int a, const std::vector<int> &v) {
    std::vector<int> res;
    for (auto value : v) {
        res.emplace_back(std::abs(std::abs(value) - std::abs(a)));
    }
    return res;
}

af::array localMaximals(const af::array &tss) {
    af::array plus = af::shift(tss, 0, -1);
    auto plusDims1 = plus.dims(1);
    plus(af::span, plusDims1 - 1, af::span) = plus(af::span, plusDims1 - 2, af::span);

    af::array minus = af::shift(tss, 0, 1);
    minus(af::span, 0, af::span) = minus(af::span, 1, af::span);

    af::array res1 = (tss > plus);
    af::array res2 = (tss > minus);
    af::array result = (res1 * res2).as(af::dtype::s32);

    return result;
}

// std::vector<LineTuple> identifyRidgeLines(const af::array &cwt_tss, const algos::array::Array<float> &maxDistances,
//                                           float gapThresh) {
//     std::vector<LineTuple> outLines;

//     // Gets all local maximals
//     af::array maximals = localMaximals(cwt_tss);
//     algos::array::Array<int> relativeMaximals(maximals);

//     // Gets all rows which contains at least one maximal
//     std::vector<int> rowsWithMaximal = algos::array::getRowsWithMaximals(relativeMaximals);
//     if (rowsWithMaximal.empty()) {
//         return outLines;
//     }

//     // Gets the last row with a maximal
//     auto startRow = rowsWithMaximal.back();

//     // Setting the first Ridge Lines (rows, cols, gap number)
//     std::vector<int> lastRowCols = algos::array::getIndexMaxColumns(relativeMaximals.getRow(startRow));
//     std::vector<CWTTuple> ridgeLines;

//     for (auto c : lastRowCols) {
//         std::vector<int> rows;
//         rows.push_back(startRow);
//         std::vector<int> cols;
//         cols.push_back(c);
//         CWTTuple newRidge = std::make_tuple(rows, cols, 0);
//         ridgeLines.push_back(newRidge);
//     }

//     // For storing the final lines
//     std::vector<CWTTuple> finalLines;

//     // Generate a range for rows
//     std::vector<int> rows;
//     for (int i = startRow - 1; i > -1; i--) {
//         rows.push_back(i);
//     }
//     // Generate a range for cols
//     std::vector<int> cols;
//     for (int i = 0; i < cwt_tss.dims(1); i++) {
//         cols.push_back(i);
//     }

//     for (const int &row : rows) {
//         std::vector<int> thisMaxCols = algos::array::getIndexMaxColumns(relativeMaximals.getRow(row));

//         // Increment the gap number of each line
//         for (auto &line : ridgeLines) {
//             std::get<2>(line) = std::get<2>(line) + 1;
//         }

//         // We store the last col for each line in prevRidgeCols
//         std::vector<int> prevRidgeCols;
//         for (auto &line : ridgeLines) {
//             prevRidgeCols.push_back(std::get<1>(line).back());
//         }

//         // Look through every relative maximum found at current row, attempt to connect them with
//         // existing ridge lines.
//         for (int col : thisMaxCols) {
//             CWTTuple *line = nullptr;
//             bool filled = false;
//             // If there is a previous ridge line within
//             // the max_distance to connect to, do so.
//             // Otherwise start a new one.
//             if (!prevRidgeCols.empty()) {
//                 std::vector<int> diffs = subsValueToVector(col, prevRidgeCols);
//                 int closest = indexMinValue(diffs);
//                 if (diffs[closest] <= maxDistances.getRow(row).front()) {
//                     line = &ridgeLines[closest];
//                     filled = true;
//                 }
//             }
//             if (filled) {
//                 // Found a point close enough, extend current ridge line
//                 std::vector<int> rows2 = std::get<0>(*line);
//                 rows2.push_back(row);
//                 std::get<0>(*line) = rows2;

//                 std::vector<int> cols2 = std::get<1>(*line);
//                 cols2.push_back(col);
//                 std::get<1>(*line) = cols2;

//                 std::get<2>(*line) = 0;
//             } else {
//                 std::vector<int> rows2;
//                 rows2.push_back(row);
//                 std::vector<int> cols2;
//                 cols2.push_back(col);
//                 CWTTuple newLine = std::make_tuple(rows2, cols2, 0);
//                 ridgeLines.push_back(newLine);
//             }
//         }

//         // Remove the ridgeLines with a gap_number too high
//         for (int ind = static_cast<int>(ridgeLines.size()) - 1; ind > -1; ind--) {
//             CWTTuple ridge = ridgeLines[ind];
//             if (std::get<2>(ridge) > gapThresh) {
//                 finalLines.push_back(ridge);
//                 ridgeLines.erase(ridgeLines.begin() + ind);
//             }
//         }
//     }

//     finalLines.insert(finalLines.end(), ridgeLines.begin(), ridgeLines.end());

//     for (auto line : finalLines) {
//         auto l = std::make_tuple(std::get<0>(line), std::get<1>(line));
//         outLines.push_back(l);
//     }

//     return outLines;
// }

float scoreAtPercentile(std::vector<float> row, int start, int end, float noisePerc) {
    std::vector<float> target;
    target.insert(target.end(), row.begin() + start, row.begin() + end);

    std::sort(target.begin(), target.end());
    float idx = noisePerc / 100.0f * (target.size() - 1);
    int i = (int)idx;
    if (i == idx) {
        return target[i];
    } else {
        int j = i + 1;
        float w1 = j - idx;
        float w2 = idx - i;
        float sumval = w1 + w2;
        float sum = target[i] * w1 + target[j] * w2;
        return sum / sumval;
    }

    return 0.0f;
}

// std::vector<LineTuple> filterFunction(const std::vector<LineTuple> &ridgeLines, std::vector<float> noises,
//                                       const algos::array::Array<float> &cwt, int minSnr, int minLength) {
//     std::vector<LineTuple> res;

//     for (auto line : ridgeLines) {
//         if (static_cast<int>(std::get<0>(line).size()) >= minLength) {
//             float snr = std::abs(cwt.getElement(std::get<0>(line).front(), std::get<1>(line).front()) /
//                                  noises[std::get<1>(line).front()]);
//             if (snr >= minSnr) {
//                 res.push_back(line);
//             }
//         }
//     }
//     return res;
// }

// std::vector<LineTuple> filterRidgeLines(const af::array &cwtDat, const std::vector<LineTuple> &ridgeLines, int minSnr,
//                                         float noisePerc) {
//     int numPoints = static_cast<int>(cwtDat.dims(1));
//     int minLength = static_cast<int>(std::ceil(cwtDat.dims(0) / 4.0));
//     int windowSize = static_cast<int>(std::ceil(numPoints / 20.0));
//     int hfWindow = windowSize / 2;
//     int odd = windowSize % 2;

//     algos::array::Array<float> cwtValues(cwtDat);

//     std::vector<float> rowOne = cwtValues.getRow(0);
//     std::vector<float> noises(rowOne.size());

//     for (int i = 0; i < static_cast<int>(rowOne.size()); i++) {
//         int windowStart = std::max(i - hfWindow, 0);
//         int windowEnd = std::min(i + hfWindow + odd, numPoints);
//         noises[i] = scoreAtPercentile(rowOne, windowStart, windowEnd, noisePerc);
//     }

//     return filterFunction(ridgeLines, noises, cwtValues, minSnr, minLength);
// }

af::array cidCeInternal(const af::array &tss) {
    auto n = tss.dims(0);
    // Calculating tss(t + 1) - tss(t) from 0 to the length of the time series minus 1
    const auto &first = tss(af::seq(std::min(1LL, n - 1), n - 1), af::span);
    const auto &second = tss(af::seq(0, std::max(0LL, n - 2)), af::span);
    af::array diff = first - second;
    // Returning the square root of the sum of squares of diff
    return af::sqrt(af::sum(diff * diff));
}


/**
 *  @brief Return a Hann window.
 *  The Hann window is a taper formed by using a raised cosine or sine-squared with ends that touch
 *  zero.
 *
 *  The Hann window is defined as:
 * \f[
 *      w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
 *              \qquad 0 \leq n \leq M-1
 * \f]
 *  The window was named for Julius von Hann, an Austrian meteorologist. It is also known as the
 *  Cosine Bell. It is sometimes erroneously referred to as
 *  the "Hanning" window, from the use of "hann" as a verb in the original
 *  paper and confusion with the very similar Hamming window.
 *
 * @param m Number of points in the output window. If zero or less, an empty array is returned.
 * @param sym When True, generates a symmetric window, for use in filter design. When
 * False, generates a periodic window, for use in spectral analysis.
 * @return af::array The window, with the maximum value normalized to 1 (though the value 1 does
 * not appear if m is even and sym is True).
 */
af::array hannWindow(int m, bool sym) {
    af::array window;
    af::array a = af::constant(0.5, 2, 1);

    if (!sym) {
        window = af::constant(0, m + 1, 1);
    } else {
        window = af::constant(0, m, 1);
    }

    double step = (2 * af::Pi) / (m);
    af::seq s(-af::Pi, af::Pi, step);
    af::array fac = s;
    for (int k = 0; k < a.dims(0); k++) {
        window += 0.5 * af::cos(k * fac);
    }
    return window(af::seq(0, m - 1));
}

/**
 *  @brief This function detrends the given time series.
 *  @param data The time series.
 *  @return af::array The detrended time timeseries.
 */
af::array detrend(const af::array &data) {
    af::array result = data - af::tile(af::mean(data, 0), static_cast<unsigned int>(data.dims(0)));
    return result;
}

/**
 * @brief This is a helper function that does the main FFT calculation. All input valdiation is performed there,
 * and the data axis is assumed to be the last axis of x. It is not designed to be called externally.
 * @param tss The given time series.
 * @param win The fft windows.
 * @param nperseg
 * @param noverlap Overlapping points.
 * @param nfft number of ffts points
 * @return af::array the result from each window is returned.
 */
af::array fftHelper(const af::array &tss, const af::array &win) {
    af::array result = af::seq(0, static_cast<double>(tss.dims(0)) - 1);
    result = detrend(result);
    result = win * result;
    result = af::fft(result, 0);

    return result(af::seq(0, static_cast<double>(result.dims(0)) / 2));
}

}  // namespace

af::array gauss::features::absEnergy(const af::array &base) {
    af::array p2 = af::pow(base, 2);
    return af::sum(p2, 0);
}

af::array gauss::features::absoluteSumOfChanges(const af::array &tss) {
    auto n = tss.dims(0);
    // Calculating tss(t + 1) - tss(t) from 0 to the length of the time series minus 1
    af::array minus =
        tss(af::seq(std::min(1LL, n - 1), n - 1), af::span) - tss(af::seq(0, std::max(0LL, n - 2)), af::span);
    // Returning the sum of absolute values of the previous operation
    return af::sum(af::abs(minus), 0);
}



// void gauss::features::aggregatedLinearTrend(const af::array &t, long chunkSize, AggregationFuncDimT aggregationFunction,
//                                             af::array &slope, af::array &intercept, af::array &rvalue,
//                                             af::array &pvalue, af::array &stderrest) {
//     // Aggregating the data using the specified chunk size and aggregation function
//     af::array aggregateResult = aggregatingOnChunks(t, chunkSize, aggregationFunction);
//     // Preparing the x vector for the linear regression. Tiling it to the number of time series
//     // contained in t
//     af::array x = af::tile(af::range(aggregateResult.dims(0)).as(t.type()), 1, static_cast<unsigned int>(t.dims(1)));
//     // Calculating the linear regression and storing the results in the parameters passed as reference
//     gauss::regression::linear(x, aggregateResult, slope, intercept, rvalue, pvalue, stderrest);
// }

// void gauss::features::aggregatedLinearTrend(const af::array &t, long chunkSize, AggregationFuncInt aggregationFunction,
//                                             af::array &slope, af::array &intercept, af::array &rvalue,
//                                             af::array &pvalue, af::array &stderrest) {
//     // Aggregating the data using the specified chunk size and aggregation function
//     af::array aggregateResult = aggregatingOnChunks(t, chunkSize, aggregationFunction);
//     // Preparing the x vector for the linear regression. Tiling it to the number of time series
//     // contained in t
//     af::array x = af::tile(af::range(aggregateResult.dims(0)).as(t.type()), 1, static_cast<unsigned int>(t.dims(1)));
//     // Calculating the linear regression and storing the results in the parameters passed as reference
//     gauss::regression::linear(x, aggregateResult, slope, intercept, rvalue, pvalue, stderrest);
// }

af::array gauss::features::approximateEntropy(const af::array &tss, int m, float r) {
    long n = static_cast<long>(tss.dims(0));
    if (r < 0) {
        throw std::invalid_argument("Parameter r must be positive ...");
    }

    if (n <= (m + 1)) {
        return 0;
    }

    return af::abs(entropy(tss, m, r) - entropy(tss, m + 1, r));
}




af::array gauss::features::binnedEntropy(const af::array &tss, int max_bins) {
    auto len = tss.dims(0);
    auto nts = tss.dims(1);
    af::array res = af::constant(0, 1, nts, tss.type());

    gfor(af::seq i, nts) {
        auto span = tss(af::span, i);
        af::array his = af::histogram(span, max_bins);
        af::array probs = his / float(len);
        af::array aux = probs * af::log(probs);
        af::array sum = af::sum(aux, 0);
        res(0, i) = sum;
    }
    return af::abs(res);
}

af::array gauss::features::c3(const af::array &tss, long lag) {
    // Product of shifting tss 2 * -lag times, with tss shifted -lag times, with the original tss
    af::array aux = af::shift(tss, 2 * static_cast<int>(-lag)) * af::shift(tss, static_cast<int>(-lag)) * tss;
    // Return the slice of the previous calculation up to the length of the time series minus 2 * lag
    return af::mean(aux(af::seq(static_cast<double>(tss.dims(0)) - 2 * lag), af::span), 0);
}

af::array gauss::features::cidCe(const af::array &tss, bool zNormalize) {
    // Apply z-normalization if specified
    if (zNormalize) {
        auto norm = gauss::normalization::znorm(tss);
        return cidCeInternal(norm);
    }
    return cidCeInternal(tss);
}

af::array gauss::features::countAboveMean(const af::array &tss) {
    af::array mean = af::mean(tss, 0);
    af::array aboveMean = (tss > af::tile(mean, static_cast<unsigned int>(tss.dims(0)))).as(af::dtype::u32);
    return af::sum(aboveMean, 0);
}

af::array gauss::features::countBelowMean(const af::array &tss) {
    // Calculating the mean of all the time series in tss
    af::array mean = af::mean(tss, 0);
    // Calculating the elements that are lower than the mean
    af::array belowMean = (tss < af::tile(mean, static_cast<unsigned int>(tss.dims(0)))).as(af::dtype::u32);
    // Sum of all elements below the mean
    return af::sum(belowMean, 0);
}

af::array gauss::features::cwtCoefficients(const af::array &tss, const af::array &widths, int coeff, int w) {
    int len = static_cast<int>(tss.dims(0));
    int nts = static_cast<int>(tss.dims(1));
    if (len < coeff) {
        return af::constant(af::NaN, 1, nts);
    }
    if (len < coeff) {
        return af::constant(af::NaN, 1, nts);
    }
    af::array output = cwt(tss, widths);

    // To find w in widths
    af::array index;
    af::array maximum;
    af::array aux = af::abs(widths - w) * (-1);
    af::max(maximum, index, aux, 0);
    // WORKAROUND: Forcing movement of index to CPU mem, just to avoid problems with Intel GPU
    auto i = index.scalar<unsigned int>();
    // Select the corresponding values of coeff and w
    return af::reorder(output(i, coeff, af::span), 0, 2, 1);
}

af::array gauss::features::energyRatioByChunks(af::array tss, long numSegments, long segmentFocus) {
    // Calculating the energy of all the time series
    af::array fullSeriesEnergy = gauss::features::absEnergy(tss);
    long n = static_cast<long>(tss.dims(0));
    // Calculating the segment length given the length of the time series and the number of segments
    long segmentLength = n / numSegments;
    // Positioning at the beginning of the segment to focus on
    long start = segmentFocus * segmentLength;
    // Calculating the end (which should be lower than the length of the time series)
    long end = std::min((segmentFocus + 1) * segmentLength, n);
    return gauss::features::absEnergy(tss(af::seq(start, end - 1), af::span)) / fullSeriesEnergy;
}

af::array gauss::features::fftAggregated(const af::array &tss) {
    af::array output;
    af::array fftComputed;
    // If n is even, the length of the transformed axis is (n/2)+1.
    // If n is odd, the length is (n + 1) / 2.
    // The output is Hermitian - symmetric, i.e.the negative frequency terms are just the complex
    // conjugates of the corresponding positive - frequency terms, and the negative - frequency
    // terms are therefore redundant
    if (tss.dims(0) % 2 == 0) {
        // number of elements is even
        int n = (static_cast<int>(tss.dims(0)) / 2) + 1;
        fftComputed = af::abs(af::fft(tss, 0))(af::seq(0, n - 1), af::span);
    } else {
        // number of elements is odd
        int n = (static_cast<int>(tss.dims(0)) + 1) / 2;
        fftComputed = af::abs(af::fft(tss, 0))(af::seq(0, n - 1), af::span);
    }
    output = af::constant(0, 4, tss.dims(1), tss.type());
    output(0, af::span) = calculateCentroid(fftComputed);
    output(1, af::span) = calculateVariance(fftComputed);
    output(2, af::span) = calculateSkew(fftComputed);
    output(3, af::span) = calculateKurtosis(fftComputed);
    return output;
}

void gauss::features::fftCoefficient(const af::array &tss, long coefficient, af::array &real, af::array &imag,
                                     af::array &abs, af::array &angle) {
    // Calculating the FFT of all the time series contained in tss
    af::array fft = af::fft(tss);
    // Slicing by the given coefficient
    af::array fftCoefficient = fft(static_cast<int>(coefficient), af::span);
    // Retrieving the real, imaginary, absolute value and angle of the complex number of the given coefficient
    real = af::real(fftCoefficient);
    imag = af::imag(fftCoefficient);
    abs = af::abs(real);
    angle = af::arg(fftCoefficient);
}

af::array gauss::features::firstLocationOfMaximum(const af::array &tss) {
    auto len = static_cast<float>(tss.dims(0));
    af::array index;
    af::array maximum;

    af::max(maximum, index, tss, 0);

    return index.as(tss.type()) / len;
}

af::array gauss::features::firstLocationOfMinimum(const af::array &tss) {
    // Flipping the array because ArrayFire returns the last location of the minimum by default
    af::array flipped = af::flip(tss, 0);

    af::array minimum;
    af::array index;

    // Calculating the minimum and the ocurring index
    af::min(minimum, index, flipped, 0);

    // Complementing the index since we flipped the array before
    index = af::abs(index - tss.dims(0) + 1);

    // Dividing by the length of the time series because the result is relative
    return index.as(tss.type()) / tss.dims(0);
}

// af::array gauss::features::friedrichCoefficients(const af::array &tss, int m, float r) {
//     return estimateFriedrichCoefficients(tss, m, r);
// }

af::array gauss::features::hasDuplicates(const af::array &tss) {
    // Array with the number of input time series in the 1st dimension of type bool
    auto result = af::array(tss.dims(1), af::dtype::b8);

    // Doing it sequentially because af::setUnique only works with vectors
    for (dim_t i = 0; i < tss.dims(1); ++i) {
        // Calculating the unique elements for each time series
        af::array uniq = af::setUnique(tss(af::span, i));
        // If the number of elements differ, then the time series has duplicates
        result(i) = tss.dims(0) != uniq.dims(0);
    }

    // Transposing the array to match the dimensions of the output
    return af::transpose(result);
}

af::array gauss::features::hasDuplicateMax(const af::array &tss) {
    af::array maximum = af::max(tss, 0);
    return af::sum(tss == af::tile(maximum, static_cast<unsigned int>(tss.dims(0))), 0) > 1;
}

af::array gauss::features::hasDuplicateMin(const af::array &tss) {
    // Calculating the minimum of each time series contained in tss
    af::array minimum = af::min(tss, 0);

    // Returning if the minimum appears more than once
    return af::sum(tss == af::tile(minimum, static_cast<unsigned int>(tss.dims(0))), 0) > 1;
}

af::array gauss::features::indexMassQuantile(const af::array &tss, float q) {
    auto len = static_cast<float>(tss.dims(0));

    af::array positives = af::abs(tss);
    af::array sums = af::sum(positives, 0);
    af::array acum = af::accum(positives, 0);
    af::array geQ =
        gauss::features::firstLocationOfMaximum((acum / af::tile(sums, static_cast<unsigned int>(len))) >= q)
            .as(tss.type());
    af::array res = ((geQ * len) + 1) / len;

    return res;
}

af::array gauss::features::kurtosis(const af::array &tss) {
    // Using the kurtosis function of the statistics namespace
    return gauss::statistics::kurtosis(tss);
}

af::array gauss::features::largeStandardDeviation(const af::array &tss, float r) {
    return af::stdev(tss, 0) > (r * (af::max(tss, 0) - af::min(tss, 0)));
}

af::array gauss::features::lastLocationOfMaximum(const af::array &tss) {
    // Flipping the array because ArrayFire returns the last location of the minimum by default
    af::array flipped = af::flip(tss, 0);

    af::array maximum;
    af::array index;

    // Calculating the maximum and the occuring index
    af::max(maximum, index, flipped, 0);

    // Complementing the index since we flipped the array before
    index = af::abs(index - tss.dims(0) + 1);

    // Dividing by the length of the time series because the result is relative
    return (index.as(tss.type()) + 1) / tss.dims(0);
}

af::array gauss::features::lastLocationOfMinimum(const af::array &tss) {
    af::array minimum;
    af::array index;

    af::min(minimum, index, tss, 0);

    return (index.as(tss.type()) + 1) / tss.dims(0);
}

af::array gauss::features::length(const af::array &tss) {
    int n = static_cast<int>(tss.dims(0));
    // Returning an array containing as many ns as the number of input time series in tss
    return af::tile(af::array(1, &n), static_cast<unsigned int>(tss.dims(1)));
}

void gauss::features::linearTrend(const af::array &tss, af::array &pvalue, af::array &rvalue, af::array &intercept,
                                  af::array &slope, af::array &stder) {
    int len = static_cast<int>(tss.dims(0));
    auto ntss = static_cast<unsigned int>(tss.dims(1));
    af::array yss = af::tile(af::range(len).as(tss.type()), 1, ntss);
    gauss::regression::linear(yss, tss, slope, intercept, rvalue, pvalue, stder);
}

af::array gauss::features::localMaximals(const af::array &tss) {
    af::array up = af::shift(tss, -1, 0);
    const int upsize = static_cast<const int>(up.dims(1));
    up(upsize - 1, af::span) = af::constant(0.0, 1, upsize);

    af::array down = af::shift(tss, 1, 0);
    down(0, af::span) = af::constant(0.0, 1, down.dims(1));

    af::array res1 = (tss > up);
    af::array res2 = (tss > down);
    af::array result = (res1 * res2).as(tss.type());

    return result;
}

af::array gauss::features::longestStrikeAboveMean(const af::array &tss) {
    // Calculating the mean of each time series contained in tss
    af::array mean = af::mean(tss, 0);
    // Checking the elements of tss that are greater than the mean
    af::array aboveMean = (tss > af::tile(mean, static_cast<unsigned int>(tss.dims(0)))).as(tss.type());

    // Doing a scan by key with the same array as the values in order to sum the consecutive 1s in the aboveMean array
    af::array scanned = af::scanByKey(aboveMean.as(af::dtype::s32), aboveMean);

    // Returning the maximum of the sum of consecutive 1s
    return af::max(scanned, 0);
}

af::array gauss::features::longestStrikeBelowMean(const af::array &tss) {
    af::array mean = af::mean(tss, 0);
    af::array belowMean = (tss < af::tile(mean, static_cast<unsigned int>(tss.dims(0)))).as(tss.type());

    af::array result = af::scanByKey(belowMean.as(af::dtype::s32), belowMean);

    return af::max(result, 0);
}

// af::array gauss::features::maxLangevinFixedPoint(const af::array &tss, int m, float r) {
//     af::array coefficients = estimateFriedrichCoefficients(tss, m, r);

//     af::array roots = gauss::polynomial::roots(coefficients);

//     return af::max(af::real(roots)).as(tss.type());
// }

af::array gauss::features::maximum(const af::array &tss) { return af::max(tss, 0); }

af::array gauss::features::mean(const af::array &tss) { return af::mean(tss, 0); }

af::array gauss::features::meanAbsoluteChange(const af::array &tss) {
    return (gauss::features::absoluteSumOfChanges(tss) / tss.dims(0)).as(tss.type());
}

af::array gauss::features::meanChange(const af::array &tss) {
    auto n = static_cast<float>(tss.dims(0));
    return af::sum(af::diff1(tss, 0), 0) / n;
}

af::array gauss::features::meanSecondDerivativeCentral(const af::array &tss) {
    auto n = tss.dims(0);
    // Calculating tss(t + 2) - 2 * tss(t + 1) + tss(t) from 0 to the length of the time series minus - 2
    af::array total = tss(af::seq(0, n - 3), af::span);
    total -= 2 * tss(af::seq(1, n - 2), af::span);
    total += tss(af::seq(2, n - 1), af::span);
    return af::sum(total, 0) / (2 * n);
}

af::array gauss::features::median(const af::array &tss) { return af::median(tss, 0); }

af::array gauss::features::minimum(const af::array &tss) { return af::min(tss, 0); }

af::array gauss::features::numberCrossingM(const af::array &tss, int m) {
    return af::sum(af::abs(af::diff1(tss > m)), 0).as(tss.type());
}

// af::array algos::features::numberCwtPeaks(const af::array &tss, int maxW) {
//     af::array out = af::constant(0, 1, tss.dims(1), tss.type());

//     af::array widths = (af::range(af::dim4(maxW)) + 1).as(af::dtype::s32);
//     float gapThresh = 1;
//     af::array max_distances = widths / 4.0;
//     algos::array::Array<float> maxDistances(max_distances);

//     // Computing one timeseries at a time, due to divergencies in the algorithm
//     for (int i = 0; i < tss.dims(1); i++) {
//         af::array cwt_tss = cwt(tss(af::span, i), widths);

//         std::vector<LineTuple> ridgeLines = identifyRidgeLines(cwt_tss, maxDistances, gapThresh);

//         std::vector<LineTuple> filtered = filterRidgeLines(cwt_tss, ridgeLines, 1, 10);

//         std::vector<int> maxLoc;
//         for (auto line : filtered) {
//             maxLoc.push_back(std::get<1>(line).front());
//         }

//         std::sort(maxLoc.begin(), maxLoc.end());
//         out(0, i) = maxLoc.size();
//     }

//     return out;
// }

af::array gauss::features::numberPeaks(af::array tss, int n) {
    auto length = tss.dims(0);
    af::array tssReduced = tss(af::seq(n, length - n - 1), af::span);

    // Initializing it to ones
    af::array res = tssReduced == tssReduced;
    for (int i = 1; i < n + 1; i++) {
        af::array rightShifted = af::shift(tss, i)(af::seq(n, length - n - 1), af::span);
        af::array leftShifted = af::shift(tss, -i)(af::seq(n, length - n - 1), af::span);
        res = res & ((tssReduced > rightShifted) & (tssReduced > leftShifted));
    }

    return af::sum(res.as(tss.type()), 0);
}

af::array gauss::features::percentageOfReoccurringDatapointsToAllDatapoints(const af::array &tss, bool isSorted) {
    af::array result = af::array(1, tss.dims(1), tss.type());
    // Doing it sequentially because the setUnique function can only be used with a vector
    for (int i = 0; i < tss.dims(1); i++) {
        af::array unique = af::setUnique(tss(af::span, i), isSorted);
        auto n = unique.dims(0);
        // The chunk size cannot be greater than the length of the unique values
        auto chunkSize = std::min(n, BATCH_SIZE);

        af::array tmpResult = af::array(0, tss.type());

        for (dim_t j = 0; j < n; j += chunkSize) {
            // The iteration space cannot be greater than what is left (n - j)
            auto iterationSize = std::min(chunkSize, n - j);
            af::array uniqueChunk = unique(af::seq(j, j + iterationSize - 1));

            af::array tssTiled = af::tile(tss(af::span, i), 1, static_cast<unsigned int>(uniqueChunk.dims(0)));
            uniqueChunk = af::transpose(af::tile(uniqueChunk, 1, static_cast<unsigned int>(tss.dims(0))));

            tmpResult = af::join(0, tmpResult, af::sum(af::sum(uniqueChunk == tssTiled, 0) > 1, 1).as(tss.type()));
        }
        result(i) = af::sum(tmpResult, 0) / (float)unique.dims(0);
    }

    return result;
}

af::array gauss::features::percentageOfReoccurringValuesToAllValues(const af::array &tss, bool isSorted) {
    af::array result = af::constant(0, 1, tss.dims(1), tss.type());
    // Doing it sequentially because the setUnique function can only be used with a vector
    for (int i = 0; i < tss.dims(1); i++) {
        af::array uniques = af::setUnique(tss(af::span, i), isSorted);
        auto n = uniques.dims(0);
        af::array tmp = af::constant(0, 1, n, tss.type());
        // Computing the number of occurrences for each unique value
        for (auto j = 0; j < n; j++) {
            tmp(0, j) = af::count(
                tss(af::span, i) == af::tile(uniques(j), static_cast<unsigned int>(tss(af::span, i).dims(0)), 1), 0);
        }
        // WORKAROUND: Because of indirect memory access fails on Intel GPU
        // result(0, i) = af::sum(tmp(0, aux), 1);
        af::array aux = af::where(tmp - 1);
        for (int h = 0; h < aux.dims(0); h++) {
            result(0, i) = result(0, i) + tmp(0, aux(h).scalar<unsigned int>());
        }
    }
    return result / tss.dims(0);
}

// af::array gauss::features::quantile(const af::array &tss, const af::array &q, float precision) {
//     return gauss::statistics::quantile(tss, q, precision);
// }

af::array gauss::features::rangeCount(const af::array &tss, float min, float max) {
    af::array mins = (tss > min).as(tss.type());
    af::array maxs = (tss < max).as(tss.type());

    return af::sum(mins * maxs, 0);
}

af::array gauss::features::ratioBeyondRSigma(const af::array &tss, float r) {
    auto n = static_cast<float>(tss.dims(0));

    af::array greaterThanRSigma = af::abs(tss - af::tile(af::mean(tss, 0), static_cast<unsigned int>(tss.dims(0)))) >
                                  af::tile(r * af::stdev(tss, 0), static_cast<unsigned int>(tss.dims(0)));

    return af::sum(greaterThanRSigma.as(tss.type()), 0) / n;
}

af::array gauss::features::ratioValueNumberToTimeSeriesLength(const af::array &tss) {
    af::array result = af::constant(0, 1, tss.dims(1), tss.type());
    for (auto i = 0; i < tss.dims(1); i++) {
        auto n = static_cast<float>(af::setUnique(tss(af::span, i)).dims(0));
        result(0, i) = n / tss.dims(0);
    }
    return result;
}

af::array gauss::features::sampleEntropy(const af::array &tss) {
    auto n = tss.dims(0);

    af::array std = af::stdev(tss, 0);
    std *= 0.2;

    const dim_t chunkSize = 32768;
    long lastIterationSize = chunkSize;

    af::array query = af::array(1, chunkSize, tss.dims(1), tss.type());
    af::array reference = af::array(1, chunkSize, tss.dims(1), tss.type());

    af::array A = af::constant(0, tss.dims(1), tss.type());
    af::array B = af::constant(0, tss.dims(1), tss.type());

    // it performs a batching (or blocking) in the horizontal direction
    for (dim_t i = 0; i < n - 1; i++) {
        // it performs a batching (or blocking) in the vertical direction
        for (dim_t j = i + 1; j < n; j += chunkSize) {
            auto iterationSize = std::min(chunkSize, n - j);
            // if our batching dimension does not match the dimension of the remaining elements, modify dimensions
            if (iterationSize != chunkSize || iterationSize != lastIterationSize) {
                lastIterationSize = iterationSize;
                query = af::array(1, iterationSize, tss.dims(1), tss.type());
                reference = af::array(1, iterationSize, tss.dims(1), tss.type());
            }

            query = af::tile(af::reorder(tss(af::seq(i, i), af::span), 0, 2, 1, 3), 1,
                             static_cast<unsigned int>(iterationSize));
            reference = af::reorder(tss(af::seq(j, j + iterationSize - 1), af::span), 2, 0, 1, 3);

            // Get the maximum difference among all dimensions for each time series
            af::array distances = af::abs(reference - query);
            // sum the number of elements bigger than the threshhold given by (stdev*r)
            af::array count =
                distances <= af::tile(af::reorder(std, 0, 2, 1, 3), static_cast<unsigned int>(distances.dims(0)),
                                      static_cast<unsigned int>(distances.dims(1)));
            // we summarise all partial sums in sum; we accumulate all partial sums for each vertical dimension
            A += af::reorder(af::sum(af::sum(count, 0), 1), 2, 0, 1, 3);
        }
    }

    float N = n * (n - 1) / 2.0f;

    B = af::tile(af::array(1, &N), static_cast<unsigned int>(tss.dims(1))).as(tss.type());

    return -af::log(A / B);
}

af::array gauss::features::skewness(const af::array &tss) { return gauss::statistics::skewness(tss); }

af::array gauss::features::spktWelchDensity(const af::array &tss, int coeff) {
    float fs = 1.0;
    int nfft = static_cast<int>(tss.dims(0));

    af::array window = hannWindow(static_cast<int>(tss.dims(0)), false);
    float scale = 1.0f / (fs * af::sum(window * window, 0).scalar<float>());

    af::array out = af::constant(0, (tss.dims(0) / 2) + 1, tss.dims(1));
    for (int i = 0; i < tss.dims(1); i++) {
        af::array result = fftHelper(tss, window);
        result = af::conjg(result) * result;
        result *= scale;
        result *= 2;
        result(0) = result(0) / 2;

        if ((nfft % 2) != 0) {
            // Last point is unpaired Nyquist freq point, don't double
            result(static_cast<const int>(result.dims(0)) - 1) = result(static_cast<const int>(result.dims(0)) - 1) / 2;
        }
        out(af::span, i) = af::real(result);
    }
    return out(coeff, af::span);
}

af::array gauss::features::standardDeviation(const af::array &tss) { return af::stdev(tss, 0); }

af::array gauss::features::sumOfReoccurringDatapoints(const af::array &tss, bool isSorted) {
    af::array result = af::array(1, tss.dims(1), tss.type());
    // Doing it sequentially because the setUnique function can only be used with a vector
    for (auto i = 0; i < tss.dims(1); i++) {
        af::array unique = af::setUnique(tss(af::span, i), isSorted);
        auto n = unique.dims(0);
        // The chunk size cannot be greater than the length of the unique values
        auto chunkSize = std::min(n, BATCH_SIZE);

        af::array counts = af::array(0, tss.type());

        for (auto j = 0; j < n; j += chunkSize) {
            // The iteration space cannot be greater than what is left (n - j)
            auto iterationSize = std::min(chunkSize, n - j);
            af::array uniqueChunk = unique(af::seq(j, j + iterationSize - 1));

            af::array tssTiled = af::tile(tss(af::span, i), 1, static_cast<unsigned int>(uniqueChunk.dims(0)));
            uniqueChunk = af::transpose(af::tile(uniqueChunk, 1, static_cast<unsigned int>(tss.dims(0))));

            counts = af::join(0, counts, af::transpose(af::sum(uniqueChunk == tssTiled, 0).as(tss.type())));
        }
        counts(counts < 2) = 0.0;
        result(i) = af::sum(counts * unique, 0);
    }

    return result;
}

af::array gauss::features::sumOfReoccurringValues(const af::array &tss, bool isSorted) {
    af::array result = af::constant(0, 1, tss.dims(1), tss.type());
    // Doing it sequentially because the setUnique function can only be used with a vector
    for (int i = 0; i < tss.dims(1); i++) {
        af::array uniques = af::setUnique(tss(af::span, i), isSorted);
        int n = static_cast<int>(uniques.dims(0));
        af::array tmp = af::constant(0, 1, n, tss.type());
        // Computing the number of occurrences for each unique value
        for (int j = 0; j < n; j++) {
            tmp(0, j) = af::count(
                tss(af::span, i) == af::tile(uniques(j), static_cast<unsigned int>(tss(af::span, i).dims(0)), 1), 0);
        }
        result(0, i) = af::sum(uniques(tmp > 1), 0);
    }
    return result;
}

af::array gauss::features::sumValues(const af::array &tss) { return af::sum(tss, 0); }

af::array gauss::features::symmetryLooking(const af::array &tss, float r) {
    af::array meanMedianAbsDifference = af::abs(af::mean(tss, 0) - af::median(tss, 0));
    af::array maxMinDifference = af::max(tss, 0) - af::min(tss, 0);
    return meanMedianAbsDifference < (r * maxMinDifference);
}

af::array gauss::features::timeReversalAsymmetryStatistic(const af::array &tss, int lag) {
    int n = static_cast<int>(tss.dims(0));
    if (n <= (2 * lag)) {
        throw std::invalid_argument("The size of tss needs to be larger for this lag value ...");
    }

    af::array l_0 = tss(af::seq(0, n - (2 * lag) - 1), af::span);
    af::array l_l = tss(af::seq(lag, n - lag - 1), af::span);
    af::array l_2l = tss(af::seq(2 * lag, n - 1), af::span);

    return af::sum((l_2l * l_2l * l_l) - (l_l * l_0 * l_0), 0) / (n - 2 * lag);
}

af::array gauss::features::valueCount(const af::array &tss, float v) {
    af::array value = af::tile(af::array(1, &v), tss.dims());

    return af::sum((value == tss).as(af::dtype::u32), 0);
}

af::array gauss::features::variance(const af::array &tss) { return af::var(tss, true, 0); }

af::array gauss::features::varianceLargerThanStandardDeviation(const af::array &tss) {
    return af::var(tss, false, 0) > af::stdev(tss, 0);
}
