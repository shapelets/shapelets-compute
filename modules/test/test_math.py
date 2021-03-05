import numpy as np
import shapelets.compute as sh


def test_sqrt():
    n = np.sqrt([[1, 4, 9, 16], [25, 36, 42, 56.]])
    a = sh.sqrt([[1, 4, 9, 16], [25, 36, 42, 56.]])
    assert a.same_as(n)
