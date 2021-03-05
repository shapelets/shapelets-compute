#ifndef AF_ARRAY_ERROR__H
#define AF_ARRAY_ERROR__H

#include <arrayfire.h>

void warn_if_error(const af_err &err);
void throw_on_error(const af_err& err);

#endif  //AF_ARRAY_ERROR__H
