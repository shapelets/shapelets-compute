/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#ifndef GAUSS_MATRIX_H
#define GAUSS_MATRIX_H

#include <arrayfire.h>
#include <gauss/defines.h>

#include <utility>
#include <vector>
#include <optional>

namespace gauss::matrix {

/**
 * @brief Calculates the N best matches of several queries in several time series.
 *
 * The result has the following structure:
 *  - 1st dimension corresponds to the nth best match.
 *  - 2nd dimension corresponds to the number of queries.
 *  - 3rd dimension corresponds to the number of time series.
 *
 * For example, the distance in the position (1, 2, 3) corresponds to the second best distance of the third query in the
 * fourth time series. The index in the position (1, 2, 3) is the is the index of the subsequence which leads to the
 * second best distance of the third query in the fourth time series.
 *
 * @param q Array whose first dimension is the length of the query time series and the second dimension is the number of
 * queries.
 * @param t Array whose first dimension is the length of the time series and the second dimension is the number of time
 * series.
 * @param n Number of matches to return.
 * @param distances Resulting distances.
 * @param indexes Resulting indexes.
 */
GAUSSAPI void findBestNOccurrences(const af::array &q, const af::array &t, long n, af::array &distances,
                                   af::array &indexes);

/**
 * @brief Mueen's Algorithm for Similarity Search.
 *
 * The result has the following structure:
 *  - 1st dimension corresponds to the index of the subsequence in the time series.
 *  - 2nd dimension corresponds to the number of queries.
 *  - 3rd dimension corresponds to the number of time series.
 *
 * For example, the distance in the position (1, 2, 3) correspond to the distance of the third query to the fourth time
 * series for the second subsequence in the time series.
 *
 * [1] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva,
 * Abdullah Mueen, Eamonn Keogh (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View
 * that Includes Motifs, Discords and Shapelets. IEEE ICDM 2016.
 *
 * @param q Array whose first dimension is the length of the query time series and the second dimension is the number of
 * queries.
 * @param t Array whose first dimension is the length of the time series and the second dimension is the number of time
 * series.
 * @param distances Resulting distances.
 */
GAUSSAPI void mass(const af::array &q, const af::array &t, af::array &distances);

/**
 * @brief This function extracts the best N motifs from a previously calculated matrix profile.
 *
 * @param profile The matrix profile containing the minimum distance of each subsequence.
 * @param index The matrix profile index containing where each minimum occurs.
 * @param m Subsequence length value used to calculate the input matrix profile.
 * @param n Number of motifs to extract.
 * @param motifs The distance of the best N motifs.
 * @param motifsIndices The indices of the best N motifs.
 * @param subsequenceIndices The indices of the query sequences that produced the minimum reported in the motifs
 * output array.
 * @param selfJoin Indicates whether the input profile comes from a self join operation or not. It determines
 * whether the mirror similar region is included in the output or not.
 */
GAUSSAPI void findBestNMotifs(const af::array &profile, const af::array &index, long m, long n, af::array &motifs,
                              af::array &motifsIndices, af::array &subsequenceIndices, bool selfJoin = false);

/**
 * @brief This function extracts the best N discords from a previously calculated matrix profile.
 *
 * @param profile The matrix profile containing the minimum distance of each subsequence.
 * @param index The matrix profile index containing where each minimum occurs.
 * @param m Subsequence length value used to calculate the input matrix profile.
 * @param n Number of discords to extract.
 * @param discords The distance of the best N discords.
 * @param discordsIndices The indices of the best N discords.
 * @param subsequenceIndices The indices of the query sequences that produced the discords reported in the discords
 * output array.
 * @param selfJoin Indicates whether the input profile comes from a self join operation or not. It determines
 * whether the mirror similar region is included in the output or not.
 */
GAUSSAPI void findBestNDiscords(const af::array &profile, const af::array &index, long m, long n, af::array &discords,
                                af::array &discordsIndices, af::array &subsequenceIndices, bool selfJoin = false);

/**
 * @brief STOMP algorithm to calculate the matrix profile between 'ta' and 'tb' using a subsequence length of 'm'.
 *
 * [1] Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah Mueen,
 * Philip Brisk and Eamonn Keogh (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
 * Hundred Million Barrier for Time Series Motifs and Joins. IEEE ICDM 2016.
 *
 * @param ta Query time series.
 * @param tb Reference time series.
 * @param m Subsequence length.
 * @param profile The matrix profile, which reflects the distance to the closer element of the subsequence from 'ta'
 * in 'tb'.
 * @param index The matrix profile index, which points to where the aforementioned minimum is located.
 */
GAUSSAPI void stomp(const af::array &ta, const af::array &tb, long m, af::array &profile, af::array &index);

/**
 * @brief STOMP algorithm to calculate the matrix profile between 't' and itself using a subsequence length of 'm'.
 * This method filters the trivial matches.
 *
 * [1] Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah Mueen,
 * Philip Brisk and Eamonn Keogh (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
 * Hundred Million Barrier for Time Series Motifs and Joins. IEEE ICDM 2016.
 *
 * @param t Query and reference time series.
 * @param m Subsequence length.
 * @param profile The matrix profile, which reflects the distance to the closer element of the subsequence from 't' in a
 * different location of itself.
 * @param index The matrix profile index, which points to where the aforementioned minimum is located.
 */
GAUSSAPI void stomp(const af::array &t, long m, af::array &profile, af::array &index);

/**
 * @brief Calculates the matrix profile between 't' and itself using a subsequence length of 'm'.
 * This method filters the trivial matches.
 *
 * [1] Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah Mueen,
 * Philip Brisk and Eamonn Keogh (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
 * Hundred Million Barrier for Time Series Motifs and Joins. IEEE ICDM 2016.
 *
 * @param tss Query time series.
 * @param m Subsequence length.
 * @param profile The matrix profile, which reflects the distance to the closer element of the subsequence from 'ta'
 * in 'tb'.
 * @param index The matrix profile index, which points to where the aforementioned minimum is located.
 */
GAUSSAPI void matrixProfile(const af::array &tss, long m, af::array &profile, af::array &index);

/**
 * @brief Calculates the matrix profile between 'ta' and 'tb' using a subsequence length of 'm'.
 *
 * [1] Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah Mueen,
 * Philip Brisk and Eamonn Keogh (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
 * Hundred Million Barrier for Time Series Motifs and Joins. IEEE ICDM 2016.
 *
 * @param ta Query and reference time series.
 * @param tb Query and reference time series.
 * @param m Subsequence length.
 * @param profile The matrix profile, which reflects the distance to the closer element of the subsequence from 't' in a
 * different location of itself.
 * @param index The matrix profile index, which points to where the aforementioned minimum is located.
 */
GAUSSAPI void matrixProfile(const af::array &ta, const af::array &tb, long m, af::array &profile, af::array &index);

/**
 * @brief Calculates the matrix profile to the left and to the right between 't' and using a subsequence length of 'm'.
 *
 * [1] Yan Zhu, Makoto Imamura, Daniel Nikovski, and Eamonn Keogh. Matrix Profile VII: Time Series Chains: A New
 * Primitive for Time Series Data Mining. IEEE ICDM 2017
 *
 * @param tss Time series to compute the matrix profile.
 * @param m Subsequence length.
 * @param profileLeft The matrix profile distance to the left.
 * @param indexLeft The subsequence index of the matrix profile to the left.
 * @param profileRight The matrix profile distance to the right.
 * @param indexRight The subsequence index of the matrix profile to the right.
 *
 *  Notice that when there is no match the subsequence index is the length of tss.
 */
GAUSSAPI void matrixProfileLR(const af::array &tss, long m, af::array &profileLeft, af::array &indexLeft,
                              af::array &profileRight, af::array &indexRight);

/**
 * @brief Calculates all the chains within 'tss' using a subsequence length of 'm'.
 *
 * [1] Yan Zhu, Makoto Imamura, Daniel Nikovski, and Eamonn Keogh. Matrix Profile VII: Time Series Chains: A New
*  Primitive for Time Series Data Mining. IEEE ICDM 2017
 *
 * @param tss Time series to compute the chains within them.
 * @param m Subsequence length.
 * @param chains The calculated chains with the following topology:
 *  - 1st dimension corresponds to the chains indexes flattened.
 *  - 2nd dimension:
                - [0] corresponds to all the indexes in the chains flattened
                - [1] corresponds to the index of the chain that the value in [0] belongs to.
 *  - 3rd dimension corresponds to the number of time series.
 *
 *  Notice that the size of the first dimension is the maximum possible size which is n - m + 1. If the number of
 *  values belonging to a chain is lower than the maximum, the remaining values and indexes are 0. It implies
 *  that 0 is an invalid chain index.
 */
GAUSSAPI void getChains(const af::array &tss, long m, af::array &chains);


GAUSSAPI af::array mpdist_vector(const af::array &tss, const af::array &ts_b, long w, double threshold = 0.05);

typedef struct {
    uint32_t index;
    uint32_t size;
    uint32_t window;
    double pct;
    af::array distances; 
    af::array indices;
} snippet_t;

GAUSSAPI std::vector<snippet_t> snippets(const af::array& tss, 
                                         const uint32_t snippet_size, 
                                         const uint32_t num_snippets, 
                                         const std::optional<uint32_t>& window_size = std::nullopt); 


GAUSSAPI af::array cac(const af::array& profile, const af::array& index, const long w);
GAUSSAPI std::vector<unsigned int> segment(const af::array& profile, const af::array& index, const unsigned int w, const int num_reg = -1, const unsigned int ez = 5);


}  // namespace gauss

#endif
