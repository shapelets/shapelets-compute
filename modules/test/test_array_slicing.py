import pytest
import numpy as np
import shapelets.compute as sh


def test_slicing_invert():
    a = sh.iota((3, 3), 1)
    assert a[::-1, ...].same_as(sh.flip(a, 0))
    assert a[..., ::-1].same_as(sh.flip(a, 1))


def test_slicing_ranges():
    a = sh.iota((4, 4), 1)
    # first two rows
    assert a[0:2, ::].same_as([[0, 4, 8, 12], [1, 5, 9, 13]])
    # every two rows
    assert a[0::2, ::].same_as([[0, 4, 8, 12], [2, 6, 10, 14]])
    # every tow rows from 1
    assert a[1::2, ::].same_as([[1, 5, 9, 13], [3, 7, 11, 15]])
    # every tow rows from 1, backwards
    assert a[::-2, ::].same_as([[3, 7, 11, 15], [1, 5, 9, 13]])
    # just one element
    assert a[0, 0].same_as(([0]))


def test_equivalence_with_numpy():
    a = sh.iota((4, 4), 1)
    n = np.array(a)
    # first two rows
    assert a[0:2, ::].same_as(n[0:2, ::])
    # every two rows
    assert a[0::2, ::].same_as(n[0::2, ::])
    # every tow rows from 1
    assert a[1::2, ::].same_as(n[1::2, ::])
    # every tow rows from 1, backwards
    assert a[::-2, ::].same_as(n[::-2, ::])
    # just one element
    assert a[0, 0].same_as(n[0, 0])
    assert a[0:2, ::-1].same_as(n[0:2, ::-1])


def test_ellipsis_operator():
    a = sh.iota((3, 3, 3), 1, dtype="int32")
    n = np.array(a)
    # we don't do squeeze by default
    assert a[...].same_as(a)
    ts = a[0, ..., 0]
    tn = n[0, ..., 0]
    ts = sh.moddims(ts, (3, 1))
    assert ts.same_as(tn)

    ts = a[::, 1, ...]
    tn = n[::, 1, ...]
    ts = sh.moddims(ts, (3, 3))
    assert ts.same_as(tn)

    assert a[..., 1, 1].same_as(n[..., 1, 1])


# TODO: IMPLEMENT SQUEEZE

def test_logical_filters():
    a = sh.iota((3, 3), 1, dtype="int32")
    assert ((a < 1) | (a > 5)).same_as([[1, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert (1 < a < 4).same_as([[1, 1, 0], [1, 0, 0], [1, 0, 0]])
    assert a[(a < 5)].same_as([0, 1, 2, 3, 4])

    # mixing boolean selectors and indexers could yield to unexpected
    # results.  For example:
    b = a[::, (a < 5)]
    # yields to
    assert b.same_as([
        [0, 3, 6, 6, 3],
        [1, 4, 7, 7, 4],
        [2, 5, 8, 8, 5]
    ])
    # because the selector is saying:
    #  - select all rows (::)
    #  - select all columns whose indices are the positions of those elements less than
    #    5 (linear indices Fortran ordering).  Since these indices are 0, 1, 2, 3, 4, it will
    #    yield columns 0, 1, 2 and 3,4 are interpreted in circular order, offsetting from the
    #    last column!

    # you can use a combination of logical filtering and element wise operation to
    # adjust the array to you your needs, but, more often than not, replace should
    # be more adequate.
    assert (a * (1 < a < 4)).same_as(sh.replace(a, 1 < a < 4, 0.))


def test_assigment_scalar():
    c = sh.constant((3, 3, 3), 0, dtype="complex64")
    c[1, ...] = 1+1j
    c[..., 1] = 1-1j

    assert c.same_as([[[0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j]],
                      [[1.+1.j, 1.-1.j, 1.+1.j],
                       [1.+1.j, 1.-1.j, 1.+1.j],
                       [1.+1.j, 1.-1.j, 1.+1.j]],
                      [[0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j]]])

    c[1, 1, 1] = -1-1j
    assert c.same_as([[[0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j]],
                      [[1.+1.j, 1.-1.j, 1.+1.j],
                       [1.+1.j, -1.-1.j, 1.+1.j],
                       [1.+1.j, 1.-1.j, 1.+1.j]],
                      [[0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j],
                       [0.+0.j, 1.-1.j, 0.+0.j]]])


def test_assigment_vector():
    c = sh.constant((3, 3, 3), 0., dtype="float32")
    d = sh.constant((3, 3, 3), 1., dtype="float32")
    c[0, ..., 0] = 1.
    c[0, ..., 0] = d[0, ..., 0]
