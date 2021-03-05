#ifndef AF_ARRAY__H
#define AF_ARRAY__H

#include "templates.h"
#include "array_obj.h"
#include "index_slicing.h"
#include "errors.h"
#include "conversions.h"
#include "numpy.h"
#include "algorithms.h"
#include "batches.h"
#include "enums.h"
#include "complex.h"
#include "construction.h"
#include "extract_transform.h"
#include "informative.h"
#include "linear_algebra.h"
#include "math.h"
#include "random.h"
#include "statistics.h"
#include "device.h"
#include "signal_processing.h"
#include "array_like.h"
#include "logic.h"

void af_array_bindings(py::module &m);

#endif  //AF_ARRAY_ERROR__H
