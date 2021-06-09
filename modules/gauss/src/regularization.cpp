/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <gauss/regularization.h>

af::array gauss::regularization::groupBy(const af::array &in, AggregationFuncDimT aggregationFunction, int nColumnsKey,
                                         int nColumnsValue) {
    auto n = in.dims(0);

    af::seq keyColumns = af::seq(nColumnsKey);
    af::seq valuesColumns = af::seq(nColumnsValue) + static_cast<double>(nColumnsKey);

    af::array minus = in(af::seq(1, n - 1), keyColumns) - in(af::seq(0, n - 2), keyColumns);
    af::array groupKeysMask = af::anyTrue(af::join(0, af::constant(1, 1, nColumnsKey, minus.type()), minus) != 0, 1);
    af::array groupKeys = in(groupKeysMask, keyColumns);
    af::array values = af::constant(0, groupKeys.dims(0), nColumnsValue, in.type());

    // GFOR cannot be used because the WHERE clause (implicit when using a mask)
    // cannot be used inside
    // gfor(af::seq i, groupKeys.dims(0)) {
    for (int i = 0; i < groupKeys.dims(0); i++) {
        af::array tiledGroupKey = af::tile(groupKeys(i, af::span), static_cast<unsigned int>(in.dims(0)));
        af::array valuesMask = af::allTrue(in(af::span, keyColumns) == tiledGroupKey, 1);
        values(i, af::span) += aggregationFunction(in(valuesMask, valuesColumns), 0);
    }

    return values;
}

af::array gauss::regularization::groupBy(const af::array &in, AggregationFuncBoolDimT aggregationFunction,
                                         int nColumnsKey, int nColumnsValue) {
    auto n = in.dims(0);

    af::seq keyColumns = af::seq(nColumnsKey);
    af::seq valuesColumns = af::seq(nColumnsValue) + (double)nColumnsKey;

    af::array minus = in(af::seq(1, n - 1), keyColumns) - in(af::seq(0, n - 2), keyColumns);
    af::array groupKeysMask = af::anyTrue(af::join(0, af::constant(1, 1, nColumnsKey, minus.type()), minus) != 0, 1);
    af::array groupKeys = in(groupKeysMask, keyColumns);
    af::array values = af::constant(0, groupKeys.dims(0), nColumnsValue, in.type());

    // GFOR cannot be used because the WHERE clause (implicit when using a mask)
    // cannot be used inside
    // gfor(af::seq i, groupKeys.dims(0)) {
    for (int i = 0; i < groupKeys.dims(0); i++) {
        af::array tiledGroupKey = af::tile(groupKeys(i, af::span), static_cast<unsigned int>(in.dims(0)));
        af::array valuesMask = af::allTrue(in(af::span, keyColumns) == tiledGroupKey, 1);
        values(i) += aggregationFunction(in(valuesMask, valuesColumns), true, 0);
    }

    return values;
}

af::array gauss::regularization::groupBy(const af::array &in, AggregationFuncInt aggregationFunction, int nColumnsKey,
                                         int nColumnsValue) {
    auto n = in.dims(0);

    af::seq keyColumns = af::seq(nColumnsKey);
    af::seq valuesColumns = af::seq(nColumnsValue) + (double)nColumnsKey;

    af::array minus = in(af::seq(1, n - 1), keyColumns) - in(af::seq(0, n - 2), keyColumns);
    af::array groupKeysMask = af::anyTrue(af::join(0, af::constant(1, 1, nColumnsKey, minus.type()), minus) != 0, 1);
    af::array groupKeys = in(groupKeysMask, keyColumns);
    af::array values = af::constant(0, groupKeys.dims(0), nColumnsValue, in.type());

    // GFOR cannot be used because the WHERE clause (implicit when using a mask)
    // cannot be used inside
    // gfor(af::seq i, groupKeys.dims(0)) {
    for (int i = 0; i < groupKeys.dims(0); i++) {
        af::array tiledGroupKey = af::tile(groupKeys(i, af::span), static_cast<unsigned int>(in.dims(0)));
        af::array valuesMask = af::allTrue(in(af::span, keyColumns) == tiledGroupKey, 1);
        values(i) += aggregationFunction(in(valuesMask, valuesColumns), 0);
    }

    return values;
}
