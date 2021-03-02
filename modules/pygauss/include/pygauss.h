#ifndef PYGAUSS__H
#define PYGAUSS__H

#include <gauss.h>
#include "af_array/af_array.h"

void gauss_matrix_bindings(py::module &m);
void gauss_statistic_bindings(py::module &m);


#endif // PYGAUSS__H