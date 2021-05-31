/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <pygauss.h>

namespace spd = spdlog;

std::string get_last_error() {
    char *msg;
    dim_t msgl;
    af_get_last_error(&msg, &msgl);
    return std::string(msg, msgl);
}

std::string build_msg(const af_err &err) {
    auto last_error = get_last_error();
    std::stringstream ss;
    ss << af_err_to_string(err) << ".  Error Code: " << err << std::endl << std::endl;
    ss << "Last Error Message:" << std::endl;
    ss << last_error << std::endl;
    return ss.str();
}

namespace pygauss {

    void warn_if_error(const af_err &err) {
        if (err == AF_SUCCESS) return;
        spd::error(build_msg(err));
    }

    void throw_on_error(const af_err &err) {
        if (err == AF_SUCCESS) return;

        auto err_msg = build_msg(err);
        spd::error(err_msg);
        throw std::runtime_error(err_msg);
    }
}
