#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <af_array/af_array.h>

namespace spd = spdlog;

void check_af_error(const af_err &err) {
    if (err != AF_SUCCESS) {
        spd::debug("Error Code is {}", err);
        std::stringstream ss;
        ss << af_err_to_string(err) << ".  Error Code: " << err;
        throw std::runtime_error(ss.str());
    }
}