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
    assert a[0, 0].same_as([0])


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
    # first three rows, all cols backwards
    assert a[0:2, ::-1].same_as(n[0:2, ::-1])
    # just one element
    assert a[0, 0].same_as(n[0, 0])
    


def test_ellipsis_operator():
    a = sh.iota((3, 3, 3), 1, dtype="int32")
    n = np.array(a)
    # we don't do squeeze by default
    assert a[...].same_as(a)
    ts = a[0, ..., 0]
    tn = n[0, ..., 0]
    ts = sh.reshape(ts, (3, 1))
    assert ts.same_as(tn)

    ts = a[::, 1, ...]
    tn = n[::, 1, ...]
    ts = sh.reshape(ts, (3, 3))
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
    assert (a * ((1 < a) & (a < 4))).same_as(sh.where(((1 < a) & (a < 4)), a, 0.))


def test_assigment_scalar():
    c = sh.zeros((3, 3, 3),dtype="complex64")
    c[1, ...] = 1 + 1j
    c[..., 1] = 1 - 1j

    assert c.same_as([[[0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j]],
                      [[1. + 1.j, 1. - 1.j, 1. + 1.j],
                       [1. + 1.j, 1. - 1.j, 1. + 1.j],
                       [1. + 1.j, 1. - 1.j, 1. + 1.j]],
                      [[0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j]]])

    c[1, 1, 1] = -1 - 1j
    assert c.same_as([[[0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j]],
                      [[1. + 1.j, 1. - 1.j, 1. + 1.j],
                       [1. + 1.j, -1. - 1.j, 1. + 1.j],
                       [1. + 1.j, 1. - 1.j, 1. + 1.j]],
                      [[0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j],
                       [0. + 0.j, 1. - 1.j, 0. + 0.j]]])


def test_assignment_vector():
    c = sh.zeros((3, 3, 3), dtype="float32")
    d = sh.ones((3, 3, 3), dtype="float32")
    c[0, ..., 0] = 1.
    c[0, ..., 0] = d[0, ..., 0]


def test_batch_lambda():
    # Batch is no longer part of the lib
    # in favour for automatic broadcasting
    assert True   
    # f = sh.array([
    #     [-0.92466235, 0.18082578, 2.544097, 0.35158235, -0.3451673]
    # ], dtype="float32")

    # w = sh.array([
    #     [0.21912338, -0.5377797, -0.61736727, -1.1410631, 1.2990667],
    #     [-0.76866555, 0.22698347, 0.58894515, 0.8281328, 0.08831034],
    #     [0.24125516, 0.93536, 0.78968173, -0.7363408, 0.65880156],
    #     [-1.1948396, -0.76128507, -0.06451859, -0.74457514, -0.27883467],
    #     [0.89269227, 0.57870644, 0.9520341, -0.8414588, 1.4143447]
    # ], dtype="float32")

    # filtered_weights = sh.batch(lambda: f * w)

    # assert filtered_weights.same_as([
    #     [-0.20261514, -0.09724443, -1.5706422, -0.40117764, -0.44839534],
    #     [0.7107561, 0.04104447, 1.4983336, 0.2911569, -0.03048184],
    #     [-0.22307956, 0.16913721, 2.0090268, -0.25888443, -0.22739676],
    #     [1.1048232, -0.13765997, -0.16414154, -0.2617795, 0.09624461],
    #     [-0.8254389, 0.10464504, 2.4220672, -0.29584205, -0.48818555]
    # ])


def test_batch_with():
    # batch is no longer part of the api
    # in favour for automatic broadcasting.
    assert True 

    # def my_batch_function(lhs, rhs):
    #     with sh.batch():
    #         return lhs * rhs

    # f = sh.array([
    #     [-0.92466235, 0.18082578, 2.544097, 0.35158235, -0.3451673]
    # ], dtype="float32")  # 1x5

    # w = sh.array([
    #     [0.21912338, -0.5377797, -0.61736727, -1.1410631, 1.2990667],
    #     [-0.76866555, 0.22698347, 0.58894515, 0.8281328, 0.08831034],
    #     [0.24125516, 0.93536, 0.78968173, -0.7363408, 0.65880156],
    #     [-1.1948396, -0.76128507, -0.06451859, -0.74457514, -0.27883467],
    #     [0.89269227, 0.57870644, 0.9520341, -0.8414588, 1.4143447]
    # ], dtype="float32")  # 5x5

    # filtered_weights = my_batch_function(f, w)
    # assert filtered_weights.same_as([
    #     [-0.20261514, -0.09724443, -1.5706422, -0.40117764, -0.44839534],
    #     [0.7107561, 0.04104447, 1.4983336, 0.2911569, -0.03048184],
    #     [-0.22307956, 0.16913721, 2.0090268, -0.25888443, -0.22739676],
    #     [1.1048232, -0.13765997, -0.16414154, -0.2617795, 0.09624461],
    #     [-0.8254389, 0.10464504, 2.4220672, -0.29584205, -0.48818555]
    # ])


def test_batch_parallel_range():
    n = 5
    m = 5
    a = sh.random.random((n, m), dtype="float32")
    b = sh.zeros((n, m), dtype="float32")

    for ii in sh.parallel_range(m):
        b[..., ii] = sh.sin(ii) + a[..., ii]

    assert (b - a).same_as([
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.8415,  0.8415,  0.8415,  0.8415,  0.8415],
        [ 0.9093,  0.9093,  0.9093,  0.9093,  0.9093],
        [ 0.1411,  0.1411,  0.1411,  0.1411,  0.1411],
        [-0.7568, -0.7568, -0.7568, -0.7568, -0.7568]
    ])

def test_assign_vector_with_boolean_condition():
    a = sh.zeros(10, "bool")
    b = sh.random.randn(10)
    a[b<0.5] = True 
    assert a.same_as(b<0.5)

def test_assign_matrix():
    a = sh.zeros((10,10), "int32")
    b = sh.array([False, False, False, False, True, False, False, False, False, False], dtype="bool")
    c = sh.array([5], 1, dtype="int32")
    a[:, b] = sh.array([1,2,3,4,5,6,7,8,9,0]).T
    a[:, c] = sh.array([1,2,3,4,5,6,7,8,9,0]).T
    assert a[:,3].same_as(sh.array([0,0,0,0,0,0,0,0,0,0]).T)
    assert a[:,4].same_as(sh.array([1,2,3,4,5,6,7,8,9,0]).T)
    assert a[:,5].same_as(sh.array([1,2,3,4,5,6,7,8,9,0]).T)
    assert a[:,6].same_as(sh.array([0,0,0,0,0,0,0,0,0,0]).T)