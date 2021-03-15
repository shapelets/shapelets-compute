#include <gauss/distances.h>
#include <gauss/normalization.h>

#include <algorithm>

class DistanceAlgorithm {
    public:
        virtual bool all_same_length() = 0;
        virtual af::array compute(const af::array& src, const af::array& dst) = 0;
};

class Hamming: DistanceAlgorithm {
    public:
        Hamming() {}
        bool all_same_length() { return true; }
        
        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, af::dtype::f32);
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sum(src != dst(af::span, ii));
            }
            return result.as(af::dtype::s32);
        }
};

class Manhattan: DistanceAlgorithm {
    public:
        Manhattan() {}
        bool all_same_length() { return true; }
        
        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sum(af::abs(src - dst(af::span, ii)));
            }
            return result;
        }
};

class Euclidian: DistanceAlgorithm {
    public:
        Euclidian() {}

        bool all_same_length() { return true; }
        
        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sqrt(af::sum(af::pow(src - dst(af::span, ii), 2)));
            }
            return result;
        }
};

class Chebyshev: DistanceAlgorithm {
    public:
        Chebyshev() {}

        bool all_same_length() { return true; }
        
        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::max(af::abs(src - dst(af::span, ii)), 1);
            }
            return result;
        }
};

class Minkowshi: DistanceAlgorithm {
    public:
        Minkowshi(double p): p(p), pinv(1.0/p) {}

        bool all_same_length() { return true; }
        
        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                auto diff = src - dst(af::span, ii);
                auto diff_p = af::pow(diff, p);
                auto sum = af::sum(diff_p);
                result(0, ii) = af::pow(sum, pinv);
            }
            return result;
        }
    private:
        double p; 
        double pinv;   
};

class SBD: DistanceAlgorithm {
    public:
        SBD() {}

        bool all_same_length() { return false; }

        af::array compute(const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            
            af::array xNorm = af::sqrt(af::sum(af::pow(src, 2), 0));
            gfor(auto ii, dst_cols) {
                auto other = dst(af::span, ii);
                af::array yNorm = af::sqrt(af::sum(af::pow(other, 2), 0));
                result(0, ii) = 1.0 - af::max(af::convolve(src, af::flip(other, 0), AF_CONV_EXPAND), 0) / (xNorm * yNorm);
            }
            return result;
        }
};



namespace {


inline double distance(double x, double y) { return std::sqrt(std::pow((x - y), 2)); }

af::array distance(const af::array &a, const af::array &bss) {
    return af::sqrt(af::pow(af::tile(a, 1, static_cast<unsigned int>(bss.dims(1))) - bss, 2));
}

af::array dtwInternal(const af::array &a, const af::array &bss) {
    auto m = a.dims(0);
    auto n = bss.dims(0);

    // Allocate the cost Matrix:
    af::array cost = af::constant(0, m, n, bss.dims(1));
    auto d = distance(a(0), bss(0, af::span));
    cost(0, 0, af::span) = af::reorder(d, 0, 2, 1, 3);

    // Calculate the first column
    for (int i = 1; i < m; i++) {
        cost(i, 0, af::span) = cost(i - 1, 0, af::span) + af::reorder(distance(a(i), bss(0, af::span)), 0, 2, 1, 3);
    }

    // Calculate the first row
    for (int j = 1; j < n; j++) {
        cost(0, j, af::span) = cost(0, j - 1, af::span) + af::reorder(distance(a(0), bss(j, af::span)), 0, 2, 1, 3);
    }

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            cost(i, j, af::span) =
                af::min(cost(i - 1, j, af::span), af::min(cost(i, j - 1, af::span), cost(i - 1, j - 1, af::span))) +
                af::reorder(distance(a(i), bss(j, af::span)), 0, 2, 1, 3);
        }
    }
    return af::reorder(cost(m - 1, n - 1, af::span), 0, 2, 1, 3);
}

}  // namespace

double gauss::distances::dtw(const std::vector<double> &t0, const std::vector<double> &t1) {
    auto m = t0.size();
    auto n = t1.size();

    // Allocate the cost Matrix
    std::vector<std::vector<double>> cost(m, std::vector<double>(n));

    cost[0][0] = distance(t0[0], t1[0]);

    // Calculate the first column
    for (size_t i = 1; i < m; i++) {
        cost[i][0] = cost[i - 1][0] + distance(t0[i], t1[0]);
    }

    // Calculate the first row
    for (size_t j = 1; j < n; j++) {
        cost[0][j] = cost[0][j - 1] + distance(t0[0], t1[j]);
    }

    // Computing the remaining values (we could apply the wavefront parallel pattern, to compute it in parallel)
    for (size_t i = 1; i < m; i++) {
        for (size_t j = 1; j < n; j++) {
            cost[i][j] =
                std::min(cost[i - 1][j], std::min(cost[i][j - 1], cost[i - 1][j - 1])) + distance(t0[i], t1[j]);
        }
    }

    return cost[m - 1][n - 1];
}

af::array gauss::distances::dtw(const af::array &tss) {
    // get the number of time series
    auto numOfTs = tss.dims(1);
    // the result is a squared matrix of dimensions numOfTs x numOfTs
    // which is initialised as zero.
    auto result = af::constant(0, numOfTs, numOfTs, tss.type());

    // for each time series, calculate in parallel all distances
    for (dim_t currentCol = 0; currentCol < numOfTs - 1; currentCol++) {
        gfor(af::seq otherCol, currentCol + 1, static_cast<double>(numOfTs - 1)) {
            auto currentSpan = tss(af::span, currentCol);
            auto otherSpan = tss(af::span, otherCol);
            result(currentCol, otherCol) = dtwInternal(currentSpan, af::reorder(otherSpan, 0, 3, 1, 2));
        }
    }

    return result;
}

af::array gauss::distances::euclidean(const af::array &tss) {
    // simply invokes non squared version and completes with
    // an elementwise sqrt operation.
    return af::sqrt(gauss::distances::squaredEuclidean(tss));
}

af::array gauss::distances::hamming(const af::array &tss) {
    // get the number of time series
    auto numOfTs = tss.dims(1);
    // the result is a squared matrix of dimensions numOfTs x numOfTs
    // which is initialised as zero.
    auto result = af::constant(0, numOfTs, numOfTs, tss.type());

    // for each time series, calculate in parallel all distances
    for (dim_t currentCol = 0; currentCol < numOfTs - 1; currentCol++) {
        gfor(af::seq otherCol, currentCol + 1, static_cast<double>(numOfTs - 1)) {
            result(currentCol, otherCol) =
                af::sum((tss(af::span, currentCol) != tss(af::span, otherCol)).as(af::dtype::s32));
        }
    }
    return result;
}




af::array hamming(const af::array& xa, const af::array &xb) {
    auto xa_len = xa.dims(1);
    auto xb_len = xb.dims(1);

    auto xa_rows = xa.dims(0);
    auto xb_rows = xb.dims(0);
    auto max_rows = std::max(xa_rows, xb_rows);
    auto checked_xa = (xa_rows < max_rows) ? 
                      af::pad(xa, af::dim4(0,0,0,0), af::dim4(max_rows-xa_rows,0,0,0), af::borderType::AF_PAD_ZERO) :
                      xa;

    auto checked_xb = (xb_rows < max_rows) ? 
                      af::pad(xb, af::dim4(0,0,0,0), af::dim4(max_rows-xb_rows,0,0,0), af::borderType::AF_PAD_ZERO) :
                      xb;

    auto result = af::array(0, xa_len, xb_len, af::dtype::s32);
    for (auto xa_col = 0; xa_col < xa_len; xa_col++) {
        auto xa_current = checked_xa(af::span, xa_col);
        gfor(auto ii, 0, xb_len-1) {
            result(xa_col, ii) = af::sum(xa_current != checked_xb(af::span, ii), 1, 0.0);
        }
    }
    return result;
}

af::array hamming(const std::vector<af::array>& xa, const std::vector<af::array>& xb) {
    auto xa_len = xa.size();
    auto xb_len = xb.size();

    auto max_xa = std::max_element(xa.begin(), xa.end(), [](const af::array& left, const af::array& right) {
            return left.dims(0) < right.dims(0);
        });
    auto max_xb = std::max_element(xb.begin(), xb.end(), [](const af::array& left, const af::array& right) {
            return left.dims(0) < right.dims(0);
        });

    auto max_rows = std::max(max_xa->dims(0), max_xb->dims(0));

    auto no_pad = af::dim4(0,0,0,0);
    auto tile_to_xb = af::dim4(1, xb_len);
    
    auto tile_to_max_rows = [=](dim_t size) { 
        return af::dim4(max_rows - size, 0, 0, 0); 
    };

    auto pad_to_max = [=](const af::array& a) { 
        return af::pad(a, no_pad, tile_to_max_rows(a.dims(0)), af::borderType::AF_PAD_ZERO);
    };

    auto xb_arr = af::array(af::dim4(max_rows, xb_len), xb[0].type());
    for (auto xb_col =0; xb_col < xb_len; xb_col++) {
        xb_arr(af::span, xb_col) = pad_to_max(xb[xb_col]);
    }

    auto result = af::array(0, xa_len, xb_len, af::dtype::s32);
    for (auto xa_col = 0; xa_col < xa_len; xa_col++) {
        auto xa_current = pad_to_max(xa[xa_col]);
        gfor(auto ii, 0, xb_len -1) {
            result(xa_col, ii) = af::sum(xa_current != xb_arr(af::span, ii), 1, 0.0);
        }
    }
    return result;
}



af::array gauss::distances::manhattan(const af::array &tss) {
    // get the number of time series
    auto numOfTs = tss.dims(1);
    // the result is a squared matrix of dimensions numOfTs x numOfTs
    // which is initialised as zero.
    auto result = af::constant(0, numOfTs, numOfTs, tss.type());

    // for each time series, calculate in parallel all distances
    for (dim_t currentCol = 0; currentCol < numOfTs - 1; currentCol++) {
        gfor(af::seq otherCol, currentCol + 1, static_cast<double>(numOfTs - 1)) {
            result(currentCol, otherCol) = af::sum(af::abs(tss(af::span, currentCol) - tss(af::span, otherCol)));
        }
    }
    return result;
}

af::array gauss::distances::sbd(const af::array &tss) {
    // get the number of time series
    auto numOfTs = tss.dims(1);
    // the result is a squared matrix of dimensions numOfTs x numOfTs
    // which is initialised as zero.
    auto result = af::constant(0, numOfTs, numOfTs, tss.type());

    // for each time series, calculate in parallel all distances
    for (dim_t currentCol = 0; currentCol < numOfTs - 1; currentCol++) {
        gfor(af::seq otherCol, currentCol + 1, numOfTs - 1) {
            af::array xZNorm = gauss::normalization::znorm(tss(af::span, currentCol));
            af::array yZNorm = gauss::normalization::znorm(tss(af::span, otherCol));
            af::array xNorm = af::sqrt(af::sum(af::pow(xZNorm, 2), 0));
            af::array yNorm = af::sqrt(af::sum(af::pow(yZNorm, 2), 0));
            result(currentCol, otherCol) =
                1.0 - af::max(af::convolve(xZNorm, af::flip(yZNorm, 0), AF_CONV_EXPAND), 0) / (xNorm * yNorm);
        }
    }

    return result;
}

af::array gauss::distances::squaredEuclidean(const af::array &tss) {
    // get the number of time series
    auto numOfTs = tss.dims(1);
    // the result is a squared matrix of dimensions numOfTs x numOfTs
    // which is initialised as zero.
    auto result = af::constant(0, numOfTs, numOfTs, tss.type());

    // for each time series, calculate in parallel all distances
    for (dim_t currentCol = 0; currentCol < numOfTs - 1; currentCol++) {
        gfor(af::seq otherCol, currentCol + 1, static_cast<double>(numOfTs - 1)) {
            result(currentCol, otherCol) = af::sum(af::pow(tss(af::span, currentCol) - tss(af::span, otherCol), 2));
        }
    }

    return result;
}
