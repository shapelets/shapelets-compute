# import shapelets.compute as sh
# import numpy as np
# import pytest
#
#
# def all_cases():
#     fns = [(sh.arcsin, np.arcsin), (sh.sin, np.sin), (sh.cos, np.cos), (sh.tan, np.tan)]
#     result = []
#
#     sh.set_backend(sh.Backend.CPU)
#     result += [pytest.param(sh.Backend.CPU, i, ii) for i in sh.get_devices() for ii in fns]
#
#     if sh.has_backend(sh.Backend.OpenCL):
#         sh.set_backend(sh.Backend.OpenCL)
#         result += [pytest.param(sh.Backend.OpenCL, i, ii) for i in sh.get_devices() for ii in fns]
#
#     if sh.has_backend(sh.Backend.CUDA):
#         sh.set_backend(sh.Backend.CUDA)
#         result += [pytest.param(sh.Backend.CUDA, i, ii) for i in sh.get_devices() for ii in fns]
#
#     return result
#
#
# @pytest.mark.parametrize("backend, device, fns", all_cases())
# def test_template(backend, device, fns):
#     sh.set_backend(backend)
#     sh.set_device(device)
#     print("\n", backend, device, fns[1])
#     cases = [np.float32]
#
#     for c in cases:
#         sh_fn, np_fn = fns
#         a = sh.random.randn(3, dtype=c)
#         sh_res = sh_fn(a)
#         np_res = np_fn(a)
#         assert sh_res.same_as(np_res), "\n" + repr(c) + " " + repr(np_fn) + "\n" \
#                                        + repr(a) + "\n" + repr(sh_res) + "\n" + repr(np_res)
