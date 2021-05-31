/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(STUB_NAME, m) {
    m.def("stub",
          []() {
              auto arr = af::constant(1.0f, 3, 3);
              arr = af::ifft(af::fft(arr));
              return af::sum<float>(af::matmul(arr, arr));
          },
          "Stub function");
}
