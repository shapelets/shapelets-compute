/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/internal/libraryInternal.h>
#include <gauss/internal/matrixInternal.h>
#include <gauss/normalization.h>
#include <gauss/matrix.h>

#include <stdexcept>
#include <iostream>
#include <optional>

namespace {
    constexpr long BATCH_SIZE_SQUARED = 2048;
    constexpr long BATCH_SIZE_B = 1024;
    constexpr long BATCH_SIZE_A = 8192;
} // namespace

namespace gauss::matrix
{
    void findBestNOccurrences(const af::array &q, const af::array &t, long n, af::array &distances, af::array &indexes)
    {
        if (n > t.dims(0) - q.dims(0) + 1)
            throw std::invalid_argument("You cannot retrieve more than (L-m+1) occurrences.");

        if (n < 1) 
            throw std::invalid_argument("You cannot retrieve less than one occurrences.");

        af::array distancesGlobal;
        gauss::matrix::mass(q, t, distancesGlobal);
        af::array sortedDistances;
        af::array sortedIndexes;
        af::sort(sortedDistances, sortedIndexes, distancesGlobal);
        indexes = sortedIndexes(af::seq(n), af::span, af::span);
        distances = sortedDistances(af::seq(n), af::span, af::span);
    }

    void findBestNMotifs(const af::array &profile, const af::array &index, long m, long n, af::array &motifs,
                         af::array &motifsIndices, af::array &subsequenceIndices, bool selfJoin) {
        internal::findBestN(profile, index, m, n, motifs, motifsIndices, subsequenceIndices, selfJoin, true);
    }

    void findBestNDiscords(const af::array &profile, const af::array &index, long m, long n, af::array &discords,
                           af::array &discordsIndices, af::array &subsequenceIndices, bool selfJoin) {
        internal::findBestN(profile, index, m, n, discords, discordsIndices, subsequenceIndices, selfJoin, false);
    }

    void stomp(const af::array &ta, const af::array &tb, long m, af::array &profile, af::array &index)
    {
        auto batchSizeSquared = library::internal::getValueScaledToMemoryDevice(
            BATCH_SIZE_SQUARED, gauss::library::internal::Complexity::CUADRATIC);
        if (tb.dims(0) > batchSizeSquared)
        {
            if (ta.dims(0) > batchSizeSquared)
            {
                // Calculates the distance and index profiles using a double batching strategy. First by the number of query
                // sequences from tb to compare simultaneously; and second, the chunk size of the reference time series ta
                return internal::stomp_batched_two_levels(ta, tb, m, batchSizeSquared, batchSizeSquared, profile, index);
            }
            else
            {
                // Calculates the distance and index profiles using a batching strategy by the number of query
                // sequences from tb to compare simultaneously
                return internal::stomp_batched(ta, tb, m, batchSizeSquared, profile, index);
            }
        }
        else
        {
            // Doing it in parallel
            return internal::stomp_parallel(ta, tb, m, profile, index);
        }
    }

    void stomp(const af::array &t, long m, af::array &profile, af::array &index)
    {
        const auto batchSizeSquared = library::internal::getValueScaledToMemoryDevice(
            BATCH_SIZE_SQUARED, gauss::library::internal::Complexity::CUADRATIC);

        const auto batchSizeB =
            library::internal::getValueScaledToMemoryDevice(BATCH_SIZE_B, gauss::library::internal::Complexity::CUADRATIC);

        const auto batchSizeA =
            library::internal::getValueScaledToMemoryDevice(BATCH_SIZE_A, gauss::library::internal::Complexity::CUADRATIC);
        if (t.dims(0) > batchSizeSquared)
        {
            // Calculates the distance and index profiles using a double batching strategy. First by the number of query
            // sequences from t to compare simultaneously; and second, the chunk size of the reference time series t
            return internal::stomp_batched_two_levels(t, m, batchSizeB, batchSizeA, profile, index);
        }
        else
        {
            // Doing it in parallel
            return internal::stomp_parallel(t, m, profile, index);
        }
    }

    void matrixProfile(const af::array &tss, long m, af::array &profile, af::array &index) {
        internal::scamp(tss, m, profile, index);
    }

    void matrixProfile(const af::array &ta, const af::array &tb, long m, af::array &profile, af::array &index) {
        internal::scamp(ta, tb, m, profile, index);
    }

    void matrixProfileLR(const af::array &tss, long m, af::array &profileLeft, af::array &indexLeft,
                         af::array &profileRight, af::array &indexRight) {
        internal::scampLR(tss, m, profileLeft, indexLeft, profileRight, indexRight);
    }

    typedef struct {
        af::array aux;
        af::array mean;
        af::array stdev;
        af::array t;
        af::array fft;
        int32_t w;
    } mass_t;

    mass_t mass_prepare(const af::array &t, const int32_t w) {
        af::array aux, mean, stdev;
        internal::meanStdev(t, aux, w, mean, stdev);
        auto fft = af::fft(t, 0);
        return {aux, mean, stdev, t, fft, w};
    }

    af::array mass_fft(const mass_t &state, const af::array &q) {
        // af::array slidingDotProduct(const af::array &q, const af::array &t) {
        auto n = state.t.dims(0);
        auto m = q.dims(0);

        // Flipping all the query sequences contained in q
        auto qr = af::flip(q, 0);
        auto qr_fft = af::fft(qr, n);
        auto Z = af::tile(state.fft, 1, 1, 1, qr.dims(3)) * af::conjg(qr_fft);
        auto z = af::real(af::ifft(Z));

        // Calculating the convolve of all the query sequences contained in qr
        // against all the time series contained in t
        // af::array qt = af::real(af::convolve(t, qr, AF_CONV_EXPAND, AF_CONV_FREQ));
        return z(af::seq(m - 1, n - 1), af::span, af::span, af::span);
    }

    // }

    af::array mass_compute(const mass_t &state, const af::array &queries, bool normalize) {
        auto qReordered = af::reorder(queries, 0, 3, 2, 1);
        auto checked_q = (normalize) ? gauss::normalization::znorm(qReordered) : qReordered;

        af::array qt = mass_fft(state, checked_q);
        af::array sum_q = af::sum(checked_q, 0);
        af::array sum_q2 = af::sum(af::pow(checked_q, 2), 0);

        // Calculate the distance and index profiles for all the combinations of query sequences and reference time series
        af::array distances;
        internal::calculateDistances(qt, state.aux, sum_q, sum_q2, state.mean, state.stdev, distances);
        return af::reorder(distances, 2, 0, 1, 3);
    }

    void mass(const af::array &q, const af::array &t, af::array &distances) {
        af::array aux, mean, stdev;
        auto qReordered = af::reorder(q, 0, 3, 2, 1);
        auto m = qReordered.dims(0);
        internal::meanStdev(t, aux, m, mean, stdev);
        internal::mass(qReordered, t, aux, mean, stdev, distances);
        distances = af::reorder(distances, 2, 0, 1, 3);
    }

    af::array mass_to_mpdist_vector(const af::array &mass, long w, double threshold) {

        // we are going to compute the column min on a sliding window of wize
        // w for each row on mass, which is going to represent the matrix profile
        // of ts_b;
        // at the same time, we are going to take subseq_num elements
        // of ts_mins to conform the abba profile

        // compute moving mins
        auto uw = af::unwrap(mass, w, 1, 1, 1);
        // all in parallel...
        auto uw_mins = af::min(uw, 0);

        // go back to the previous arragement...
        auto mins = af::wrap(uw_mins, mass.dims(0) - w + 1, mass.dims(1), 1, 1, 1, 1).T();

        // compute windows of ts_mins
        // by computing the min of each row (this is a column vector)
        // we have the matrix profile distances of ts to ts_b
        af::array ts_mins = af::min(mass, 1);
        auto unw_ts_mins = af::unwrap(ts_mins, w, 1, 1, 1);

        // join
        auto data = af::join(0, mins, unw_ts_mins);

        // sort
        auto sorted = sort(data, 0);
        //auto sorted = data;

        // choose
        auto dist_loc = static_cast<dim_t>(std::ceil(threshold * data.dims(0)));

        // return
        return sorted(dist_loc, af::span).T();
    }

    af::array mpdist_vector(const af::array &tss, const af::array &ts_b, long w, double threshold) {
        // Batch all the querys and run in a single operation
        // all the computations
        auto queries = af::unwrap(ts_b, w, 1, 1, 1);

        // matrix is going to be a len(tss) - w + 1 rows by len(ts_b) - w + 1 columns
        // each row represents a position of ts and on each column we have the
        // distance to each subquery.
        af::array mass;
        gauss::matrix::mass(queries, tss, mass);

        return mass_to_mpdist_vector(mass, w, threshold);
    }

    // snippet_size, num_snippets=2, window_size = None
    std::vector<snippet_t> snippets(const af::array &tss, const uint32_t snippet_size, const uint32_t num_snippets, const std::optional<uint32_t> &window_size) {
        auto n = tss.dims(0);
        auto w = window_size.value_or(snippet_size >> 1);

        if (num_snippets < 1)
            throw std::invalid_argument("At least one snippet is required.");

        if (snippet_size < 4)
            throw std::invalid_argument("Snippet sizes should be at least 4.");

        if (n < (snippet_size << 1))
            throw std::invalid_argument("Time series is too short for the snippet size.");

        if (w >= snippet_size)
            throw std::invalid_argument("Window size should be strictly less than snippet size");

        // pad end of time series with zeros
        auto num_zeros = std::floor(snippet_size * std::ceil(n / static_cast<double>(snippet_size)) - n);
        auto tss_padded = af::pad(tss, af::dim4(0, 0, 0, 0), af::dim4(num_zeros, 0, 0, 0), af::borderType::AF_PAD_ZERO);
        auto groups = tss_padded.dims(0) / snippet_size;
        n = tss_padded.dims(0);

        auto mass_state = mass_prepare(tss_padded, w);

        std::vector<af::array> distances;
        for (auto i = 0; i < n; i += snippet_size) {
            auto ts_b = tss_padded(af::seq(i, i + snippet_size - 1));

            // Batch all the querys and run in a single operation
            // all the computations
            auto queries = af::unwrap(ts_b, w, 1, 1, 1);
            auto mass = mass_compute(mass_state, queries, true);
            auto dist_vector = mass_to_mpdist_vector(mass, w, 0.05);
            // auto dist_vector = mpdist_vector(tss_padded, tss_padded(af::seq(i, i+snippet_size-1)), w);
            distances.push_back(dist_vector);
        }

        // Do a first pass so all variables and first snippet are initialised
        // with ease
        std::vector<snippet_t> results;

        for (auto it = distances.begin(); it != distances.end(); it++)
            af::eval(*it);

        auto minims = std::numeric_limits<double>::infinity();
        uint32_t index = 0;
        auto j = 0;
        for (auto it = distances.begin(); it != distances.end(); it++, j++) {
            auto s = af::sum<double>(*it);
            if (minims > s) {
                minims = s;
                index = j;
            }
        }
        results.push_back({index, snippet_size, w});

        auto minis = distances[index];
        for (auto sn = 1; sn < num_snippets; sn++) {
            auto minims = std::numeric_limits<double>::infinity();
            index = 0;
            j = 0;
            for (auto it = distances.begin(); it != distances.end(); it++, j++) {
                auto s = af::sum<double>(af::min(*it, minis));
                if (minims > s) {
                    minims = s;
                    index = j;
                }
            }

            minis = af::min(distances[index], minis);
            results.push_back({index, snippet_size, w});
        }

        for (auto it = results.begin(); it != results.end(); it++) {
            auto d = distances[(*it).index];
            auto mask = d <= minis;
            (*it).distances = d;
            (*it).pct = af::sum<double>(mask) / tss.dims(0);
            (*it).indices = af::where(mask);
        }

        return results;
    }

    af::array cac(const af::array &profile, const af::array &index, const long w) {
        auto pos = af::iota(index.dims(), af::dim4(1, 1, 1, 1), index.type());
        auto smll = af::sort(af::min(pos, index));
        auto large = af::sort(af::max(pos, index));
        af::array si, sv;
        af::sumByKey(si, sv, smll, af::constant(1, smll.dims(), index.type()));

        af::array li, lv;
        af::sumByKey(li, lv, large, af::constant(1, large.dims(), index.type()));

        auto mark = af::constant(0, index.dims(), index.type());
        mark(si) += sv;
        mark(li) -= lv;

        auto cross_count = af::accum(mark);

        auto i = af::iota(cross_count.dims());
        auto l = cross_count.dims(0);
        auto adj = 2.0 * i * (l - i) / l;

        auto normalized_cross_count = af::min(cross_count / adj, 1.0);

        normalized_cross_count(af::seq(0, w)) = 1.0;
        normalized_cross_count(af::seq(l - w - 1, l - 1)) = 1.0;

        return normalized_cross_count;
    }

    std::vector<unsigned int> segment(const af::array &profile, const af::array &index, const unsigned int w, const int num_reg, const unsigned int ez) {
        auto cacv = cac(profile, index, w);
        auto exczone = static_cast<int>(w * ez);
        auto cacv_size = static_cast<int>(cacv.dims(0));

        std::vector<unsigned int> result;
        auto exit = false;
        auto left = num_reg;
        while (!exit) {
            double real, imag;
            unsigned int idx;
            auto aferr = af_imin_all(&real, &imag, &idx, cacv.get());
            if (aferr != AF_SUCCESS) {
                std::stringstream ss;
                ss << "Unable to compute min index.  Error Code: " << aferr;
                throw std::runtime_error(ss.str());
            }

            if (std::abs(real - 1.0) > 1e-8) {
                result.push_back(idx);
                auto idx_signed = static_cast<int>(idx);
                auto exc_start = std::max(0, idx_signed - exczone);
                auto exc_end = std::min(cacv_size, idx_signed + exczone) - 1;
                cacv(af::seq(exc_start, exc_end)) = 1.0;
                left -= 1;
                exit = num_reg > 0 && left == 0;
            }
            else {
                exit = true;
            }
        }

        return result;
    }

    void getChains(const af::array &tss, long m, af::array &chains) { internal::getChains(tss, m, chains); }

    /**
    * Computes rolling mean and standard deviation
    */
    void compute_mean_std(af::array &means, af::array &stds, const af::array &t, const int32_t m) {
        auto windows = af::unwrap(t, m, 1, 1, 1);
        means = af::mean(windows, 0);
        stds = af::stdev(windows, 0);
    }

} // namespace gauss
