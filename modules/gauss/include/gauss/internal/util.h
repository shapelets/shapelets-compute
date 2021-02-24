#ifndef GAUSS_UTIL_H
#define GAUSS_UTIL_H

#ifndef BUILDING_GAUSS
#error Internal headers cannot be included from user code
#endif

#include <arrayfire.h>
#include <string>

namespace gauss::util {

std::string khiva_file_path(const std::string &path);

}  // namespace gauss

#endif