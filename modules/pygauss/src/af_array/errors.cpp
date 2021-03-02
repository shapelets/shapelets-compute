#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <af_array/af_array.h>

namespace spd = spdlog;

std::string get_last_error() {
    char* msg;
    dim_t msgl;
    af_get_last_error(&msg, &msgl);
    return std::string(msg, msgl);
}

void check_af_error(const af_err &err) {
    if (err == AF_SUCCESS) return;

    auto last_error = get_last_error();

    std::stringstream ss;
    ss << af_err_to_string(err) << ".  Error Code: " << err << std::endl << std::endl;
    ss << "Last Error Message:" << std::endl;
    ss << last_error << std::endl;

    auto err_msg = ss.str();
    spd::error(err_msg);
    throw std::runtime_error(err_msg);
}
