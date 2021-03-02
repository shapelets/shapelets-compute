import pyperf
import shapelets as sh

sh.set_backend(sh.Backend.CPU)
data = sh.random.randn(10000, dtype="float64")

runner = pyperf.Runner()
runner.timeit(name="test", setup="lambda x: x; sh.matrixprofile(data, 100)")


