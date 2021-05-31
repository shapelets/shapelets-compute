/* Copyright (c) 2021 Grumpy Cat Software S.L.
 *
 * This Source Code is licensed under the MIT 2.0 license.
 * the terms can be found in  LICENSE.md at the root of
 * this project, or at http://mozilla.org/MPL/2.0/.
 */

#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pygauss.h"

namespace py = pybind11;

struct DeviceInfo {
    int id;
    std::string name;
    std::string platform;
    std::string toolkit;
    std::string compute;
    bool isHalfAvailable;
    bool isDoubleAvailable;
};

struct DeviceMem {
    size_t bytes;
    size_t buffers;
    size_t lock_bytes;
    size_t lock_buffers;
};

void pygauss::bindings::device_operations(py::module &m) {

    py::enum_<af::Backend>(m, "Backend", "Defines the different computational backends where computations are executed")
            .value("Default", af::Backend::AF_BACKEND_DEFAULT,
                   "It would resolve to the first available backend out of CUDA, CPU and OpenCL.")
            .value("CPU", af::Backend::AF_BACKEND_CPU, "Uses CPU multicore capabilities.")
            .value("CUDA", af::Backend::AF_BACKEND_CUDA, "Executes algorithms in CUDA devices.")
            .value("OpenCL", af::Backend::AF_BACKEND_OPENCL, "Executes algorithms in OpenCL devices.")
            .export_values();

    m.def("get_backend",
          []() {
              return af::getActiveBackend();
          });

    m.def("set_backend",
          [](af::Backend newBackend) {
              af::setBackend(newBackend);
          });

    m.def("has_backend",
          [](af::Backend test_backend) {
              auto code = af::getAvailableBackends();
              return code & test_backend;
          },
          py::arg("test_backend").none(false));

    m.def("get_available_backends",
          []() {
              std::vector<af::Backend> result;
              auto code = af::getAvailableBackends();
              if (code & af::Backend::AF_BACKEND_CUDA) result.push_back(af::Backend::AF_BACKEND_CUDA);
              if (code & af::Backend::AF_BACKEND_CPU) result.push_back(af::Backend::AF_BACKEND_CPU);
              if (code & af::Backend::AF_BACKEND_OPENCL) result.push_back(af::Backend::AF_BACKEND_OPENCL);
              return result;
          });

    py::class_<DeviceInfo>(m, "DeviceInfo")
            .def_readonly("id", &DeviceInfo::id)
            .def_readonly("name", &DeviceInfo::name)
            .def_readonly("platform", &DeviceInfo::platform)
            .def_readonly("compute", &DeviceInfo::compute)
            .def_readonly("isHalfAvailable", &DeviceInfo::isHalfAvailable)
            .def_readonly("isDoubleAvailable", &DeviceInfo::isDoubleAvailable)
            .def("__repr__", [](const DeviceInfo &dev) {
                std::stringstream result;
                result << "[" << dev.id << "] ";
                result << dev.name << " (";
                result << dev.platform << " - ";
                result << dev.compute << " - ";
                result << dev.toolkit << ") ";
                result << "F64: " << (dev.isDoubleAvailable ? "✓" : "✘") << " - "
                       << "F16: " << (dev.isHalfAvailable ? "✓" : "✘");
                return result.str();
            });

    py::class_<DeviceMem>(m, "DeviceMemory")
            .def_readonly("bytes", &DeviceMem::bytes)
            .def_readonly("buffers", &DeviceMem::buffers)
            .def_readonly("locked_bytes", &DeviceMem::lock_bytes)
            .def_readonly("locked_buffers", &DeviceMem::lock_buffers)
            .def("__repr__", [](const DeviceMem &dm) {
                std::stringstream result;
                result << "bytes: " << dm.bytes << ", ";
                result << "buffers: " << dm.buffers << ", ";
                result << "locked_bytes: " << dm.lock_bytes << ", ";
                result << "locked_buffers: " << dm.lock_buffers;
                return result.str();
            });

    m.def("get_devices",
          []() {
              char b_name[64];
              char b_platform[64];
              char b_toolkit[64];
              char b_compute[64];
              auto current = af::getDevice();
              auto deviceCount = af::getDeviceCount();
              std::vector<DeviceInfo> result;
              for (int i = 0; i < deviceCount; i++) {
                  af::setDevice(i);
                  af::deviceInfo(b_name, b_platform, b_toolkit, b_compute);
                  result.push_back({
                                        i, std::string(b_name), std::string(b_platform),
                                        std::string(b_toolkit), std::string(b_compute),
                                        af::isHalfAvailable(i), af::isDoubleAvailable(i)
                                   });
              }
              af::setDevice(current);
              return result;
          });

    m.def("get_device",
          []() {
              char b_name[64];
              char b_platform[64];
              char b_toolkit[64];
              char b_compute[64];
              auto currentDevice = af::getDevice();
              af::deviceInfo(b_name, b_platform, b_toolkit, b_compute);

              return DeviceInfo{
                      currentDevice, std::string(b_name),
                      std::string(b_platform), std::string(b_toolkit),
                      std::string(b_compute), af::isHalfAvailable(currentDevice),
                      af::isDoubleAvailable(currentDevice)
              };
          });

    m.def("set_device",
          [](const std::variant<int, DeviceInfo> &dev) {
              auto currentDevice = af::getDevice();
              int query = currentDevice;
              if (auto pid = std::get_if<int>(&dev)) {
                  query = *pid;
              } else if (auto pinfo = std::get_if<DeviceInfo>(&dev)) {
                  query = pinfo->id;
              }
              if (query != currentDevice) {
                  if (query >= af::getDeviceCount())
                      throw std::runtime_error("Incorrect device id");
                  af::setDevice(query);
                  return true;
              }
              return false;
          });

    m.def("device_gc",
          []() {
              af::deviceGC();
          });

    m.def("sync",
          [](const std::optional<std::variant<int, DeviceInfo>> &dev) {
              if (!dev.has_value()) {
                  af::sync();
                  return;
              }

              int deviceId;
              auto v = dev.value();
              if (auto pid = std::get_if<int>(&v))
                  deviceId = *pid;
              else
                  deviceId = std::get<DeviceInfo>(v).id;

              af::sync(deviceId);
          },
          py::arg("dev") = py::none());

    m.def("get_device_memory",
          [](const std::optional<std::variant<int, DeviceInfo>> &dev) {
              size_t bytes, buffers, lock_bytes, lock_buffers = -1;

              auto currentDevice = af::getDevice();
              int query = currentDevice;

              if (dev.has_value()) {
                  auto v = dev.value();
                  if (auto pid = std::get_if<int>(&v)) {
                      query = *pid;
                  } else if (auto pinfo = std::get_if<DeviceInfo>(&v)) {
                      query = pinfo->id;
                  }
              }

              if (query != currentDevice) {
                  if (query >= af::getDeviceCount())
                      throw std::runtime_error("Incorrect device id");
                  af::setDevice(query);
              }

              af::deviceMemInfo(&bytes, &buffers, &lock_bytes, &lock_buffers);

              if (query != currentDevice) {
                  af::setDevice(currentDevice);
              }

              return DeviceMem{bytes, buffers, lock_bytes, lock_buffers};
          },
          py::arg("dev") = py::none());
}
