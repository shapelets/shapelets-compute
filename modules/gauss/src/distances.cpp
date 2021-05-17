#include <gauss/distances.h>
#include <gauss/normalization.h>
#include <gauss/matrix.h>

#include <algorithm>
#include <iostream>

#ifdef _MSC_VER
    #define forceinline __forceinline
#elif defined(__GNUC__)
    #define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
    #if __has_attribute(__always_inline__)
        #define forceinline inline __attribute__((__always_inline__))
    #else
        #define forceinline inline
    #endif
#else
    #define forceinline inline
#endif

// Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions 
// http://www.fisica.edu.uy/~cris/teaching/Cha_pdf_distances_2007.pdf

namespace gauss::distances {

// Macro to ease the registration of algorithms:
// -> ALGO: Public name
// -> FN  : One (col) to One (col) distance logic
// -> SYMM: Is simmetric?
#define LOCK_STEP_DST_ALGORITHM(ALGO, FN, SYMM)                           \
    distance_algorithm_t ALGO() {                                         \
        return {                                                          \
            true,                                                         \
            SYMM,                                                         \
            std::nullopt,                                                 \
            [](const af::array& src, const af::array& dst) {              \
                auto dst_cols = dst.dims(1);                              \
                auto result = af::array(1, dst_cols, src.type());         \
                gfor(auto ii, dst_cols) {                                 \
                    result(0, ii) = FN(src, dst(af::span, ii));           \
                }                                                         \
                return result;                                            \
            }                                                             \
        };                                                                \
    }                                                                          

//
// The L1 Family: gower, sorensen, soergel, kulczynski, lorentzian, canberra
//
forceinline af::array gower_one_to_one(const af::array &p, const af::array &q) {
    return (1.0/p.dims(0)) * af::sum(af::abs(p - q));
}
LOCK_STEP_DST_ALGORITHM(gower, gower_one_to_one, true)

forceinline af::array sorensen_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p - q)) / af::sum(p + q);
}
LOCK_STEP_DST_ALGORITHM(sorensen, sorensen_one_to_one, true)

forceinline af::array soergel_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p - q)) / af::sum(af::max(p, q));
}
LOCK_STEP_DST_ALGORITHM(soergel, soergel_one_to_one, true)

forceinline af::array kulczynski_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p-q) / af::sum(af::min(p,q)));
}
LOCK_STEP_DST_ALGORITHM(kulczynski, kulczynski_one_to_one, true)

forceinline af::array lorentzian_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::log1p(af::abs(p - q)));
}
LOCK_STEP_DST_ALGORITHM(lorentzian, lorentzian_one_to_one, true)

forceinline af::array canberra_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p - q) / (p + q));
}
LOCK_STEP_DST_ALGORITHM(canberra, canberra_one_to_one, true)


//
// The Intersection family: intersection, wavehedges, czekanowski,
//                          tanimoto, ruzicka, motyka
//
forceinline af::array intersection_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::min(p, q));
}
LOCK_STEP_DST_ALGORITHM(intersection, intersection_one_to_one, true)

forceinline af::array wavehedges_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(1.0 - af::min(p, q) / af::max(p, q));
}
LOCK_STEP_DST_ALGORITHM(wavehedges, wavehedges_one_to_one, true)

forceinline af::array czekanowski_one_to_one(const af::array &p, const af::array &q) {
    return 2.0 * af::sum(af::min(p, q)) / af::sum(p + q);
}
LOCK_STEP_DST_ALGORITHM(czekanowski, czekanowski_one_to_one, true)

forceinline af::array tanimoto_one_to_one(const af::array &p, const af::array &q) {
    auto sp = af::sum(p);
    auto sq = af::sum(q);
    auto smin = af::sum(af::min(p,q));
    return (sp + sq - 2.0 * smin) / (sp + sq - smin);
}
LOCK_STEP_DST_ALGORITHM(tanimoto, tanimoto_one_to_one, true)

forceinline af::array ruzicka_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::min(p, q)) / af::sum(af::max(p, q));
}
LOCK_STEP_DST_ALGORITHM(ruzicka, ruzicka_one_to_one, true)

forceinline af::array motyka_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::min(p, q)) / af::sum(p + q);
}
LOCK_STEP_DST_ALGORITHM(motyka, motyka_one_to_one, true)


//
// The Squared L2 family: squared_euclidean, pearson, neyman, squared_chi,
//                        prob_symmetric_chi, divergence, clark and
//                        additive_symm_chi
//
forceinline af::array squared_euclidean_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(p - q, 2.0));
}
LOCK_STEP_DST_ALGORITHM(squared_euclidean, squared_euclidean_one_to_one, true)

forceinline af::array pearson_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(p - q, 2.0) / q);
}
LOCK_STEP_DST_ALGORITHM(pearson, pearson_one_to_one, false)

forceinline af::array additive_symm_chi_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(p - q, 2.0) * (p + q) / (p * q));
}
LOCK_STEP_DST_ALGORITHM(additive_symm_chi, additive_symm_chi_one_to_one, true)

forceinline af::array squared_chi_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(p - q, 2.0) / (p + q));
}
LOCK_STEP_DST_ALGORITHM(squared_chi, squared_chi_one_to_one, true)

forceinline af::array prob_symmetric_chi_one_to_one(const af::array &p, const af::array &q) {
    return 2.0 * af::sum(af::pow(p - q, 2.0) / (p + q));
}
LOCK_STEP_DST_ALGORITHM(prob_symmetric_chi, prob_symmetric_chi_one_to_one, true)

forceinline af::array divergence_one_to_one(const af::array &p, const af::array &q) {
    return 2.0 * af::sum(af::pow(p - q, 2.0) / af::pow(p + q, 2.0));
}
LOCK_STEP_DST_ALGORITHM(divergence, divergence_one_to_one, true)

forceinline af::array clark_one_to_one(const af::array &p, const af::array &q) {
    return af::sqrt(af::sum(af::pow(af::abs(p - q), 2.0) / (p + q)));
}
LOCK_STEP_DST_ALGORITHM(clark, clark_one_to_one, true)

forceinline af::array neyman_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(p - q, 2.0) / p);
}
LOCK_STEP_DST_ALGORITHM(neyman, neyman_one_to_one, false)



//
// The Inner Product family: innerproduct, harmonic_mean, cosine, kumarhassebrook,
//                           jaccard and dice
//

forceinline af::array harmonic_mean_one_to_one(const af::array &p, const af::array &q) {
    return 2.0 * af::sum(p*q/(p+q));
}
LOCK_STEP_DST_ALGORITHM(harmonic_mean, harmonic_mean_one_to_one, true)

forceinline af::array innerproduct_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(p*q);
}
LOCK_STEP_DST_ALGORITHM(innerproduct, innerproduct_one_to_one, true)

forceinline af::array kumarhassebrook_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(p*q)/(af::sum(af::pow(p, 2.0))+af::sum(af::pow(q, 2.0))-af::sum(p*q));
}
LOCK_STEP_DST_ALGORITHM(kumarhassebrook, kumarhassebrook_one_to_one, true)

forceinline af::array cosine_one_to_one(const af::array &p, const af::array &q) {
    auto pt = af::sqrt(af::sum(af::pow(p, 2.0)));
    auto qt = af::sqrt(af::sum(af::pow(q, 2.0)));
    return af::sum(p*q)/(pt*qt);
}
LOCK_STEP_DST_ALGORITHM(cosine, cosine_one_to_one, true)

forceinline af::array dice_one_to_one(const af::array &p, const af::array &q) {
    return 1.0 - (af::sum(af::pow(p - q, 2.0)) / af::sum(af::pow(p, 2.0) + af::pow(q, 2.0)));
}
LOCK_STEP_DST_ALGORITHM(dice, dice_one_to_one, true)

forceinline af::array jaccard_one_to_one(const af::array &p, const af::array &q) {
    return 1.0 - (af::sum(af::pow(p - q, 2.0)) / af::sum(af::pow(p, 2.0) + af::pow(q, 2.0) - p*q));
}
LOCK_STEP_DST_ALGORITHM(jaccard, jaccard_one_to_one, true)



//
// The Fidelity family: fidelity, bhattacharyya, hellinger, matusita and square_chord
//
forceinline af::array fidelity_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::sqrt(p * q));
}
LOCK_STEP_DST_ALGORITHM(fidelity, fidelity_one_to_one, true)

forceinline af::array bhattacharyya_one_to_one(const af::array &p, const af::array &q) {
    return -af::log(af::sum(af::sqrt(p * q)));
}
LOCK_STEP_DST_ALGORITHM(bhattacharyya, bhattacharyya_one_to_one, true)

forceinline af::array matusita_one_to_one(const af::array &p, const af::array &q) {
    return af::sqrt(2.0 - (2.0 * af::sum(af::sqrt(p * q))));
}
LOCK_STEP_DST_ALGORITHM(matusita, matusita_one_to_one, true)

forceinline af::array hellinger_one_to_one(const af::array &p, const af::array &q) {
    return 2.0 * af::sqrt(1.0 - (af::sum(af::sqrt(p * q))));
}
LOCK_STEP_DST_ALGORITHM(hellinger, hellinger_one_to_one, true)

forceinline af::array square_chord_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::pow(af::sqrt(p) - af::sqrt(q), 2.0));
}
LOCK_STEP_DST_ALGORITHM(square_chord, square_chord_one_to_one, true)


//
//The Shannon’s Entropy family: kullback, jeffrey, topsoe, jensen_shannon,
//                              jensen_difference and k_divergence
//
forceinline af::array kullback_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(p * af::log(p/q));
}
LOCK_STEP_DST_ALGORITHM(kullback, kullback_one_to_one, false)

forceinline af::array jeffrey_one_to_one(const af::array &p, const af::array &q) {
    return af::sum((p-q) * af::log(p/q));
}
LOCK_STEP_DST_ALGORITHM(jeffrey, jeffrey_one_to_one, false)

forceinline af::array topsoe_one_to_one(const af::array &p, const af::array &q) {
    auto logpq = af::log(p + q);
    return af::sum(p * (af::log(2.0*p) - logpq) + q * (af::log(2.0 * q) - logpq));
}
LOCK_STEP_DST_ALGORITHM(topsoe, topsoe_one_to_one, true)

forceinline af::array k_divergence_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(p * af::log((2.0*p) / (p+q)));
}
LOCK_STEP_DST_ALGORITHM(k_divergence, k_divergence_one_to_one, false)

forceinline af::array jensen_difference_one_to_one(const af::array &p, const af::array &q) {
    auto pqh = (p+q) / 2.0;
    return af::sum(((p * af::log(p) + q * log(q)) / 2.0 )-(pqh * log(pqh)));
}
LOCK_STEP_DST_ALGORITHM(jensen_difference, jensen_difference_one_to_one, true)

forceinline af::array jensen_shannon_one_to_one(const af::array &p, const af::array &q) {
    auto logpq = af::log(p+q);
    return 0.5 * af::sum(p * (af::log(2.0*p) - logpq) + q * (af::log(2.0*q) - logpq));
}
LOCK_STEP_DST_ALGORITHM(jensen_shannon, jensen_shannon_one_to_one, true)

//
// The Combinations family: taneja, kumar_johnson and avg_l1_linf
//
forceinline af::array taneja_one_to_one(const af::array &p, const af::array &q) {
    auto pqh = (p + q) / 2.0;
    return af::sum(pqh * (af::log(pqh) - af::log(af::sqrt(p * q))));
}
LOCK_STEP_DST_ALGORITHM(taneja, taneja_one_to_one, true)

forceinline af::array kumar_johnson_one_to_one(const af::array &p, const af::array &q) {
    auto diffsq = af::pow(af::pow(p, 2.0) - af::pow(q, 2.0), 2.0);
    auto threetwo = 2.0 * af::pow(p*q, 3.0/2.0);
    return af::sum(diffsq / threetwo);
}
LOCK_STEP_DST_ALGORITHM(kumar_johnson, kumar_johnson_one_to_one, true)

forceinline af::array avg_l1_linf_one_to_one(const af::array &p, const af::array &q) {
    auto abs_diff = af::abs(p - q);
    return (af::sum(abs_diff) +  af::max(abs_diff)) / 2.0;
}
LOCK_STEP_DST_ALGORITHM(avg_l1_linf, avg_l1_linf_one_to_one, true)


//
// The Vicissitude family: vicis_wave_hedges, min_symmetric_chi, max_symmetric_chi
//
forceinline af::array vicis_wave_hedges_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p - q) / af::min(p, q));
}
LOCK_STEP_DST_ALGORITHM(vicis_wave_hedges, vicis_wave_hedges_one_to_one, true)

forceinline af::array min_symmetric_chi_one_to_one(const af::array &p, const af::array &q) {
    auto pqds = af::pow(p - q, 2.0);
    return af::min(af::sum(pqds / p), af::sum(pqds / q));
}
LOCK_STEP_DST_ALGORITHM(min_symmetric_chi, min_symmetric_chi_one_to_one, true)

forceinline af::array max_symmetric_chi_one_to_one(const af::array &p, const af::array &q) {
    auto pqds = af::pow(p - q, 2.0);
    return af::max(af::sum(pqds / p), af::sum(pqds / q));
}
LOCK_STEP_DST_ALGORITHM(max_symmetric_chi, max_symmetric_chi_one_to_one, true)



//
// The Minkowski family: euclidean, manhattan, chebyshev and minkowski
//
forceinline af::array manhattan_one_to_one(const af::array &p, const af::array &q) {
    return af::sum(af::abs(p - q));
}
LOCK_STEP_DST_ALGORITHM(manhattan, manhattan_one_to_one, true)

forceinline af::array chebyshev_one_to_one(const af::array &p, const af::array &q) {
    return af::max(af::abs(p - q));
}
LOCK_STEP_DST_ALGORITHM(chebyshev, chebyshev_one_to_one, true)

forceinline af::array euclidean_one_to_one(const af::array &p, const af::array &q) {
    return af::sqrt(af::sum(af::pow(p - q, 2.0)));
}
LOCK_STEP_DST_ALGORITHM(euclidean, euclidean_one_to_one, true)

distance_algorithm_t minkowski(double p) {
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


distance_algorithm_t mpdist(int32_t w, double threshold) {
    return {
        false,               // all same length
        true,               // is symmetric
        std::nullopt,       // results will be always integers.
        [=](const af::array& src, const af::array& dst) {
            auto dst_cols = dst.dims(1);
            auto result = af::array(1, dst_cols, src.type());
            af::array pab, pba, iab, iba;

            for (auto ii = 0; ii < dst_cols; ii++) {
                auto target = dst(af::span, ii);
                gauss::matrix::matrixProfile(src, target, w, pab, iab);
                gauss::matrix::matrixProfile(target, src, w, pba, iba);
                auto abba = af::join(0, pab, pba);
                auto sorted = af::sort(abba);
                auto upper_idx = static_cast<dim_t>(std::ceil(threshold * (target.dims(0)+ src.dims(0)))) - 1;
                auto checked_idx = std::min(sorted.dims(0)-1, upper_idx);
                result(0, ii) = sorted(checked_idx);
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

af::array _dtw_cost_distance(const af::array &a, const af::array &bss) {
    return af::sqrt(af::pow(af::tile(a, 1, static_cast<unsigned int>(bss.dims(1))) - bss, 2));
}

af::array _dtw(const af::array &a, const af::array &bss) {
    auto m = a.dims(0);
    auto n = bss.dims(0);

    // Allocate the cost Matrix:
    af::array cost = af::constant(0, m, n, bss.dims(1));
    auto d = _dtw_cost_distance(a(0), bss(0, af::span));
    cost(0, 0, af::span) = af::reorder(d, 0, 2, 1, 3);

    // Calculate the first column
    for (int i = 1; i < m; i++) {
        cost(i, 0, af::span) = cost(i - 1, 0, af::span) + af::reorder(_dtw_cost_distance(a(i), bss(0, af::span)), 0, 2, 1, 3);
    }

    // Calculate the first row
    for (int j = 1; j < n; j++) {
        cost(0, j, af::span) = cost(0, j - 1, af::span) + af::reorder(_dtw_cost_distance(a(0), bss(j, af::span)), 0, 2, 1, 3);
    }

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            cost(i, j, af::span) =
                af::min(cost(i - 1, j, af::span), af::min(cost(i, j - 1, af::span), cost(i - 1, j - 1, af::span))) +
                af::reorder(_dtw_cost_distance(a(i), bss(j, af::span)), 0, 2, 1, 3);
        }
    }
    return af::reorder(cost(m - 1, n - 1, af::span), 0, 2, 1, 3);
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
                result(0, otherCol) = _dtw(src, af::reorder(otherSpan, 0, 3, 1, 2));
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

    // prepare the result array, which is going to be (xa_len,xa_len)
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
            
            // call the computation
            auto partial = algo.compute(current_col, col_selection);

            // store horizontally
            result(xa_col, af::seq(xa_col + 1, xa_len-1)) = partial;
            
            // store vertically
            result(af::seq(xa_col + 1, xa_len-1), xa_col) = af::moddims(partial, partial.dims(1), 1);
        }
    }
    else {
        // The output is going to be simmilar to the case 
        // where the algorithm is simmetric, but we need 
        // to run all against all.  
        // d_ii is also computed to avoid logic that 
        // impedes performance and/or not to assume a 
        // result value.

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
    
    // adjust the length of the vectors if the algo requires 
    // all of them with same length 
    if (algo.same_length) {
        auto max_rows = std::max(xa_rows, xb_rows);
        if (xa_rows < max_rows) 
            checked_xa = af::pad(xa, af::dim4(0,0,0,0), af::dim4(max_rows-xa_rows,0,0,0), af::borderType::AF_PAD_ZERO);
        else if (xb_rows < max_rows) 
            checked_xb = af::pad(xb, af::dim4(0,0,0,0), af::dim4(max_rows-xb_rows,0,0,0), af::borderType::AF_PAD_ZERO);
    }

    // build the result type taking into account 
    // the preferences of the algorithm.  The 
    // geometry of the result will be (xa_len, xb_len)
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

