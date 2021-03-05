
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
