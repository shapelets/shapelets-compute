# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import pyperf
import shapelets_compute as sh

sh.set_backend(sh.Backend.CPU)
data = sh.random.randn(10000, dtype="float64")

runner = pyperf.Runner()
runner.timeit(name="test", setup="lambda x: x; sh.matrixprofile(data, 100)")
