
#ifndef ALGOS_CORE_REGULARIZATION_H
#define ALGOS_CORE_REGULARIZATION_H

#include <arrayfire.h>
#include <algos/defines.h>

namespace algos {

namespace regularization {

using AggregationFuncDimT = af::array (*)(const af::array &, const dim_t);
using AggregationFuncBoolDimT = af::array (*)(const af::array &, bool, const dim_t);
using AggregationFuncInt = af::array (*)(const af::array &, const int);

/**
 * @brief Group by operation in the input array using nColumnsKey columns as group keys and nColumnsValue columns as
 * values. The data is expected to be sorted. The aggregation function determines the operation to aggregate the values.
 *
 * @param in Input array containing the keys and values to operate with.
 * @param aggregationFunction This param determines the operation aggregating the values.
 * @param nColumnsKey Number of columns conforming the key.
 * @param nColumnsValue Number of columns conforming the value (they are expected to be consecutive to the column keys).
 *
 * @return af::array Array with the values of the group keys aggregated using the aggregationFunction.
 */
ALGOSAPI af::array groupBy(const af::array &in, AggregationFuncBoolDimT aggregationFunction, int nColumnsKey = 1,
                           int nColumnsValue = 1);

/**
 * @brief Group by operation in the input array using nColumnsKey columns as group keys and nColumnsValue columns as
 * values. The data is expected to be sorted. The aggregation function determines the operation to aggregate the values.
 *
 * @param in Input array containing the keys and values to operate with.
 * @param aggregationFunction This param determines the operation aggregating the values.
 * @param nColumnsKey Number of columns conforming the key.
 * @param nColumnsValue Number of columns conforming the value (they are expected to be consecutive to the column keys).
 *
 * @return af::array Array with the values of the group keys aggregated using the aggregationFunction.
 */
ALGOSAPI af::array groupBy(const af::array &in, AggregationFuncInt aggregationFunction, int nColumnsKey = 1,
                           int nColumnsValue = 1);

/**
 * @brief Group by operation in the input array using nColumnsKey columns as group keys and nColumnsValue columns as
 * values. The data is expected to be sorted. The aggregation function determines the operation to aggregate the values.
 *
 * @param in Input array containing the keys and values to operate with.
 * @param aggregationFunction This param determines the operation aggregating the values.
 * @param nColumnsKey Number of columns conforming the key.
 * @param nColumnsValue Number of columns conforming the value (they are expected to be consecutive to the column keys).
 *
 * @return af::array Array with the values of the group keys aggregated using the aggregationFunction.
 */
ALGOSAPI af::array groupBy(const af::array &in, AggregationFuncDimT aggregationFunction, int nColumnsKey = 1,
                           int nColumnsValue = 1);

}  // namespace regularization
}  // namespace algos

#endif
