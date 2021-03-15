#include <gauss/distances.h>
#include <gauss/normalization.h>

#include <algorithm>
#include <iostream>

namespace gauss::distances {



distance_algorithm_t hamming() {
    return { 
        true,               // all same length
        true,               // is symmetric
        af::dtype::s32,     // results will be always integers.
        [](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, af::dtype::f32);
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sum(src != dst(af::span, ii));
            }
            return result.as(af::dtype::s32);
        }
    };
}

distance_algorithm_t manhattan() {
    return { 
        true,               // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sum(af::abs(src - dst(af::span, ii)));
            }
            return result;
        }
    };
}

distance_algorithm_t chebyshev() {
    return { 
        true,               // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::max(af::abs(src - dst(af::span, ii)), 0);
            }
            return result;
        }
    };
}

distance_algorithm_t minkowshi(double p) {
    return { 
        true,               // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [=](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                auto diff = af::abs(src - dst(af::span, ii));
                auto diff_p = af::pow(diff, p);
                auto sum = af::sum(diff_p);
                result(0, ii) = af::pow(sum, 1.0/p);
            }
            return result;
        }
    };
}

distance_algorithm_t sbd() {
    return { 
        false,              // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [](const af::array& src, const af::array& dst) {
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
}



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

distance_algorithm_t euclidian() {
    return { 
        true,               // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto ii, dst_cols) {
                result(0, ii) = af::sqrt(af::sum(af::pow(src - dst(af::span, ii), 2)));
            }
            return result;
        }
    };
}

distance_algorithm_t dtw() {
    return { 
        false,              // all same length
        true,               // is symmetric
        std::nullopt,       // no preference on the result type
        [](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            gfor(auto otherCol, dst_cols) {
                auto otherSpan = dst(af::span, otherCol);
                result(0, otherCol) = dtwInternal(src, af::reorder(otherSpan, 0, 3, 1, 2));
            }
            return result;
        }
    };
}



/**
 * Runs the algo for every column in xa to all the others.
 * if the algorithm is symmetric, it will only do half of the work
 */ 
af::array compute(const distance_algorithm_t& algo, const af::array& xa) {
    // number of columns in xa
    auto xa_len = xa.dims(1);

    af::array result = af::constant(0.0, xa_len, xa_len, algo.resultType.value_or(xa.type()));
    if (algo.is_symmetric) {

        // the output is going to be something like:
        //
        //  0  d01 d02 d03
        // d01  0  d12 d13
        // d02 d12  0  d23
        // d03 d13 d23  0
        // 
        // since the computation is symmetric dij = dji 

        for (auto xa_col = 0; xa_col < (xa_len-1); xa_col++) {
            // select all rows but only those columns from the 
            // current point of iteration to the end.
            auto col_selection = xa(af::span, af::seq(xa_col+1, xa_len-1));
            auto current_col = xa(af::span, xa_col);
            
            auto partial = algo.compute(current_col, col_selection);

            // store horizontally
            result(xa_col, af::seq(xa_col + 1, xa_len-1)) = partial;
            
            // store vertically if required
            result(af::seq(xa_col + 1, xa_len-1), xa_col) = af::moddims(partial, partial.dims(1), 1);
        }
    }
    else {
        // The output is going to be simmilar to the case 
        // where the algorithm is simmetric, but we need 
        // to run all against all.  
        // I am actually leaving the computation of dii 

        // note we are going the full interval [0..xa_len(
        for (auto xa_col = 0; xa_col < xa_len; xa_col++) {
            auto current_col = xa(af::span, xa_col);
            // full row assigment.
            result(xa_col, af::span) = algo.compute(current_col, xa);
        }
    }

    return result;
};


/**
 * Runs algo for every column in xa against all columns in xb
 */ 
af::array compute(const distance_algorithm_t& algo, const af::array& xa, const af::array &xb) {

    // number of columns in xa
    auto xa_len = xa.dims(1);

    // number of columns in xb
    auto xb_len = xb.dims(1);

    // If there are more columns in xa than xb and 
    // the algorithm is symmetric, run the operation 
    // swapping the parameters so we get better speedup
    // in the inner loop.  When done, return the 
    // traspose 
    if (algo.is_symmetric && xa_len > xb_len) {
        return compute(algo, xb, xa).T();
    }

    // number of points for the series in xa
    auto xa_rows = xa.dims(0);

    // number of points for the series in xb
    auto xb_rows = xb.dims(0);

    auto checked_xa = xa;
    auto checked_xb = xb; 
    if (algo.same_length) {
        auto max_rows = std::max(xa_rows, xb_rows);
        if (xa_rows < max_rows) 
            checked_xa = af::pad(xa, af::dim4(0,0,0,0), af::dim4(max_rows-xa_rows,0,0,0), af::borderType::AF_PAD_ZERO);
        else if (xb_rows < max_rows) 
            checked_xb = af::pad(xb, af::dim4(0,0,0,0), af::dim4(max_rows-xb_rows,0,0,0), af::borderType::AF_PAD_ZERO);
    }

    // build the result type taking into account 
    // the preferences of the algorithm
    auto result = af::array(xa_len, xb_len, algo.resultType.value_or(xa.type()));

    // Run the algorithm sequentially for every column in xa...
    for (auto xa_col = 0; xa_col < xa_len; xa_col++) {
        auto xa_current = checked_xa(af::span, xa_col);
        // ... against all xb simultaneously (if possible with a gfor loop)
        result(xa_col, af::span) = algo.compute(xa_current, checked_xb);
    }

    // return the result
    return result;
}

}

