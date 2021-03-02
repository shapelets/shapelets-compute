import pytest
import numpy as np
import shapelets.compute as sh


def test_explicit_list_creation():
    sa = sh.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float64")
    na = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float64")
    assert sa.same_as(na)


def test_explicit_tuple_creation():
    sa = sh.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype="float64")
    na = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype="float64")
    assert sa.same_as(na)


def test_create_iota():
    a = sh.iota((5, 3), dtype="float32")
    assert a.same_as([[0., 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]])
    b = sh.iota((5, 3), (1, 2), dtype="float32")
    assert b.same_as([
        [0., 5, 10, 0, 5, 10],
        [1, 6, 11, 1, 6, 11],
        [2, 7, 12, 2, 7, 12],
        [3, 8, 13, 3, 8, 13],
        [4, 9, 14, 4, 9, 14]
    ])


def test_create_constant():
    a = sh.constant((5, 4, 3), 1.0, dtype="int32")
    b = np.ones((5, 4, 3), dtype="int32")
    a.same_as(b)


def test_dimension_conversion():
    a = np.random.randn(5, 4, 3, 2)
    b = sh.array(a)
    assert a.ndim == b.ndim
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    b.same_as(a)


def test_diagonal_creation():
    ones = sh.constant(4, 1.0, dtype="float32")
    diag_zero = sh.diag(ones, 0, False)
    assert diag_zero.same_as([
        [1., 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    diag_plus_one = sh.diag(ones, 1, False)
    assert diag_plus_one.same_as([
        [0., 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]])
    diag_minus_one = sh.diag(ones, -1, False)
    assert diag_minus_one.same_as(
        [[0., 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0]])


def test_diagonal_extraction():
    a = sh.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    d = sh.diag(a, 0, True)
    assert d.same_as([1, 7])


def test_identity_creation():
    i = sh.identity((5, 3), dtype="int16")
    assert i.same_as([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])


def test_range_creation():
    # Generates an array of [0, 4] along first dimension
    r1 = sh.range(5)
    assert r1.same_as([0, 1, 2, 3, 4])
    # Generates an array of [0, 4] along first dimension, tiled along second dimension
    r2 = sh.range((5, 2))
    assert r2.same_as([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    # Generates an array of [0, 2] along second dimension, tiled along first dimension
    r3 = sh.range((5, 3), 1)
    assert r3.same_as([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ])


def test_lower_upper():
    a = sh.randu((5, 5), dtype="float32")
    lower = sh.lower(a, True)
    upper = sh.upper(a, False)
    assert (lower + upper - sh.identity((5, 5))).same_as(a)


def test_pad_creation():
    a = sh.iota((3, 2), dtype="float32") + 10.
    # add one extra row at the beginning (1, 0, ...) and one at the
    # end (1, 0, 0, 0).  Values are zero.
    pz = sh.pad(a, (1, 0, 0, 0), (1, 0, 0, 0), sh.BorderType.Zero)
    assert pz.same_as([
        [0., 0],
        [10, 13],
        [11, 14],
        [12, 15],
        [0, 0]
    ])

    # add one extra column at the beginning, one extra row at the end
    # and out of bound values are clamped to the edge
    zce = sh.pad(a, (0, 1, 0, 0), (1, 0, 0, 0), sh.BorderType.ClampEdge)
    assert zce.same_as([
        [10., 10, 13],
        [11, 11, 14],
        [12, 12, 15],
        [12, 12, 15]
    ])

    # same as before but cycle out of bound values are mapped to range
    # of the dimension in cyclic fashion
    zcc = sh.pad(a, (0, 1, 0, 0), (1, 0, 0, 0), sh.BorderType.Periodic)
    assert zcc.same_as([
        [13., 10, 13],
        [14, 11, 14],
        [15, 12, 15],
        [13, 10, 13]
    ])

    # Out of bound values are symmetric over the edge
    zcs = sh.pad(a, (1, 1, 0, 0), (1, 1, 0, 0), sh.BorderType.Symmetric)
    assert zcs.same_as([
        [10, 10, 13, 13],
        [10, 10, 13, 13],
        [11, 11, 14, 14],
        [12, 12, 15, 15],
        [12, 12, 15, 15]
    ])


def test_moddims():
    # start with a single column, 12 rows
    a = sh.iota(12, 1)
    # in order, organise it as 2 rows, 6 columns
    b = sh.moddims(a, (2, 6))
    assert b.same_as([[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])
    # now 3 rows, 4 columns
    c = sh.moddims(a, (3, 4))
    assert c.same_as([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    # etc...
    d = sh.moddims(a, (4, 3))
    assert d.same_as([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]])
    e = sh.moddims(a, (6, 2))
    assert e.same_as([[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]])
    f = sh.moddims(a, (1, 12))
    assert f.same_as(a.T)


def test_flat():
    a = sh.iota((5, 5, 5), 1)
    assert sh.flat(a).same_as(sh.iota(5 * 5 * 5, 1))


def test_flip():
    a = sh.iota((3, 2), 1)
    assert a.same_as([[0, 3], [1, 4], [2, 5]])
    # flip columns
    b = sh.flip(a, 1)
    assert b.same_as([[3, 0], [4, 1], [5, 2]])
    # flip rows
    c = sh.flip(a, 0)
    assert c.same_as([[2, 5], [1, 4], [0, 3]])


def test_reorder():
    a = sh.iota((3, 3), 1)
    # rows per columns, columns per rows
    b = sh.reorder(a, 1, 0)
    b.same_as([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # one row, 3 columns, 3 depth where
    # depth is now what it used to be columns, columns are rows
    c = sh.reorder(a, 2, 0, 1)
    assert c.ndim == 3
    assert c.shape == (1, 3, 3)
    assert c.same_as([[[0, 3, 6], [1, 4, 7], [2, 5, 8]]])


def test_replace():
    a = sh.iota((3, 3), 1)
    # replaces those that are NOT a < 4
    sh.replaceInPlace(a, a < 4, -1.0)
    assert a.same_as([[0, 3, -1], [1, -1, -1], [2, -1, -1]])

    a = sh.iota((3, 3), 1)
    sh.replaceInPlace(a, a < 4, sh.constant((3, 3), -1))
    assert a.same_as([[0, 3, -1], [1, -1, -1], [2, -1, -1]])


def test_shift():
    a = sh.iota((3, 3), 1)
    # get columns as a whole and shift them to the right in circular manner
    b = sh.shift(a, 0, 1)
    assert b.same_as([[6, 0, 3], [7, 1, 4], [8, 2, 5]])
    # same, but to the left
    c = sh.shift(a, 0, -1)
    assert c.same_as([[3, 6, 0], [4, 7, 1], [5, 8, 2]])


def test_tile():
    # one simple column vector
    a = sh.iota(4, 1)
    # tile the number of rows twice
    b = sh.tile(a, 2)
    assert b.same_as([0, 1, 2, 3, 0, 1, 2, 3])
    # tile b by two columns
    c = sh.tile(b, 1, 2)
    assert c.same_as([[0, 0], [1, 1], [2, 2], [3, 3], [0, 0], [1, 1], [2, 2], [3, 3]])
    # same in one operation
    d = sh.tile(a, (2, 2))
    assert d.same_as(c)


def test_transpose():
    a = sh.iota((3, 3), 1)
    b = sh.transpose(a)
    sh.transposeInPlace(a)
    assert b.same_as(a)
    assert b.same_as([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


def test_numpy_interface_s_n_s():
    a = sh.iota((3, 5, 7, 11), 1)
    n = np.array(a)
    assert n.shape == a.shape
    assert n.dtype == a.dtype
    assert n.ndim == a.ndim
    c = sh.array(n)
    assert c.shape == a.shape
    assert c.dtype == a.dtype
    assert c.ndim == a.ndim
    assert c.same_as(a)


def test_numpy_interface_n_s_n():
    a = np.linspace(0., 100, 100, False, dtype="float32")
    a.shape = (4, 25)
    b = sh.array(a)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.ndim == b.ndim
    c = np.array(b)
    assert np.allclose(a, c)


@pytest.mark.xfail(raises=NotImplementedError)
def test_memory_view():
    sh.set_backend(sh.Backend.CPU)
    a = sh.iota((3, 5, 7, 11), 1, dtype="float32")
    b = memoryview(a)
    assert b.shape == a.shape
    assert b.f_contiguous
    assert not b.readonly
    assert b.ndim == a.ndim
    assert b.contiguous
    assert not b.c_contiguous
    # set a row to -1
    # but it doesn't work with memory views as it is not implemented
    b[0:3:] = -1.0
    a[::, 0, 0, 0].same_as([-1, -1, -1])


def test_join():
    a = sh.array([1, 2, 3, 4])
    b = sh.array([5, 6, 7, 8])
    c = sh.array([9, 10, 11, 12])
    # join as rows
    assert sh.join([a, b, c], 0).same_as([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # join on columns
    assert sh.join([a, b, c], 1).same_as([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])
