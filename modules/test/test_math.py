# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import numpy as np
import shapelets_compute.compute as sc


def test_sqrt():
    n = np.sqrt([[1, 4, 9, 16], [25, 36, 42, 56.]])
    a = sc.sqrt([[1, 4, 9, 16], [25, 36, 42, 56.]])
    assert a.same_as(n)


def test_absolute():
    a = sc.array([-1.0, 1.0])
    assert sc.absolute(a).same_as([1.0, 1.0])
    b = sc.array([3 + 4j, 3 - 4j])
    assert sc.absolute(b).same_as([5, 5])


def test_angle():
    a = sc.array([1.0, -1.0])
    assert sc.angle(a).same_as([0, 0])
    b = sc.array([1 + 1j, 1 - 1j, 0 + 0j])
    assert sc.angle(b, True).same_as([45, -45, 0])


def test_tanh():
    a = sc.array([1.0, -1.0])
    assert sc.tanh(a).same_as([0.7616, -0.7616])
    b = sc.array([1 + 1j, 1 - 1j, 0 + 0j])
    assert sc.tanh(b).same_as([1.0839 + 0.2718j, 1.0839 - 0.2718j, 0 + 0j])


def test_arccos():
    a = sc.array([1.0, -1.0])
    assert sc.arccos(a).same_as([0, 3.14159265])
    b = sc.array([1 + 1j, 1 - 1j, 0 + 0j])
    assert sc.arccos(b).same_as([0.9046 - 1.0613j, 0.9046 + 1.0613j, 1.5708 + 0j])


def test_hypot():
    left = sc.array([3., 4])
    right = sc.array([4., 3])
    assert sc.hypot(left, right).same_as([5, 5])
    assert sc.hypot(left, 4.).same_as([5, 5.65685424949238])
    assert sc.hypot(4., right).same_as([5.65685424949238, 5])
    assert sc.hypot([[3., 4], [4, 3]], [[4., 3], [3, 4]]).same_as([[5., 5], [5, 5]])


def test_arctan2():
    y = sc.array([1., 1, -1, -1])
    x = sc.array([1., 2, 1, 2])
    assert sc.arctan2(y, x).same_as([0.7854, 1.1071, 2.3562, 2.0344])


def test_arith_bcast():
    x = sc.array([1, 2, 3])
    y = sc.array([1, 2, 3, 4]).T

    (x + y).display()
    (y + x).display()

    xx = np.array([1, 2, 3])
    yy = np.array([1, 2, 3, 4]).T
    print(np.add(xx[:, np.newaxis], yy))
