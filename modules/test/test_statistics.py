import shapelets.compute as sc
import numpy as np
from shapelets.compute.statistics import XCorrScale


def test_stats_skewness_kurtosis_moments():
    rng = sc.random.random_engine()
    # using 64 bits as 32 bits arrays start carrying way 
    # too much acc errors...
    test_data = rng.exponential(shape=(10000000, 1), dtype="float64")
    # using exact solution for exponential distribution
    result = sc.statistics.skewness(test_data)
    assert result.same_as([2.0], eps=0.02)
    assert result.shape == (1, 1)
    result = sc.statistics.kurtosis(test_data)
    assert result.same_as([6.0], eps=0.02)
    assert result.shape == (1, 1)
    # since scale is 1.0 -> lambda is also 1.0
    # then, exponential moments are n!, where
    # n is the moment number
    m1 = sc.statistics.moment(test_data, 1)
    m2 = sc.statistics.moment(test_data, 2)
    m3 = sc.statistics.moment(test_data, 3)
    m4 = sc.statistics.moment(test_data, 4)
    assert m1.same_as(1.0, eps=0.02)
    assert m2.same_as(2.0, eps=0.02)
    assert m3.same_as(6.0, eps=0.02)
    assert m4.same_as(24.0, eps=0.03)

def test_stats_correlation():
    data = [[1., 2, 3, 4],
            [2., 3, 4, 5],
            [3., 4, 5, 6],
            [7., 8, 9, 0]]

    npr = np.corrcoef(data, rowvar=False)
    sr = sc.statistics.corrcoef(data)
    assert sr.same_as(npr)


def test_stats_covariance():
    data = [[1., 2, 3, 4],
            [2., 3, 4, 5],
            [3., 4, 5, 6],
            [7., 8, 9, 0]]

    npr = np.cov(data, rowvar=False)
    sr = sc.statistics.cov(data)
    assert sr.same_as(npr)


def test_stats_cross_correlation():
    a = sc.array([[1., 2, 3, 4], [2., 3, 4, 5]]).T
    b = sc.array([[3., 4, 5, 6], [7., 8, 9, 0]]).T

    # verified with octave
    (indices, corr) = sc.statistics.xcorr(a, b, None, sc.statistics.XCorrScale.NoScale)
    assert indices.same_as([-3, -2, -1, 0, 1, 2, 3.])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([6, 17, 32, 50, 38, 25, 12])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([0, 9, 26, 50, 74, 53, 28])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([12, 28, 47, 68, 50, 32, 15])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([0, 18, 43, 74, 98, 68, 35])

    (indices, corr) = sc.statistics.xcorr(a, b, None, sc.statistics.XCorrScale.Biased)
    assert indices.same_as([-3, -2, -1, 0, 1, 2, 3.])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([1.5, 4.25, 8.0, 12.5, 9.5, 6.25, 3.0])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([0.0, 2.25, 6.5, 12.5, 18.5, 13.25, 7.0])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([3.0, 7.0, 11.75, 17.0, 12.5, 8.0, 3.75])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([0.0, 4.5, 10.75, 18.5, 24.5, 17.0, 8.75])

    (indices, corr) = sc.statistics.xcorr(a, b, None, sc.statistics.XCorrScale.Unbiased)
    assert indices.same_as([-3, -2, -1, 0, 1, 2, 3.])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([6.0, 8.5, 10.6667, 12.50, 12.6667, 12.50, 12.0])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([0.0, 4.5, 8.6667, 12.50, 24.6667, 26.5, 28.0])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([12.0, 14.0, 15.6667, 17.0, 16.6667, 16.0, 15.0])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([0.0, 9.0, 14.3333, 18.50, 32.6667, 34.0, 35.0])

    (indices, corr) = sc.statistics.xcorr(a, b, None, sc.statistics.XCorrScale.Coeff)
    assert indices.same_as([-3, -2, -1, 0, 1, 2, 3.])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([0.1181, 0.3347, 0.6300, 0.9844, 0.7481, 0.4922, 0.2362])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([0.0000, 0.1180, 0.3408, 0.6554, 0.9700, 0.6947, 0.3670])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([0.1761, 0.4109, 0.6897, 0.9978, 0.7337, 0.4696, 0.2201])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([0.0000, 0.1759, 0.4201, 0.7230, 0.9575, 0.6644, 0.3420])

    # verified with octave
    (indices, corr) = sc.statistics.xcorr(a, b, 2, sc.statistics.XCorrScale.NoScale)
    assert indices.same_as([-2, -1, 0, 1, 2.])
    corr.display()
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([17, 32, 50, 38, 25])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([9, 26, 50, 74, 53])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([28, 47, 68, 50, 32])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([18, 43, 74, 98, 68])

    (indices, corr) = sc.statistics.xcorr(a, b, 2, sc.statistics.XCorrScale.Biased)
    assert indices.same_as([-2, -1, 0, 1, 2])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([4.25, 8.0, 12.5, 9.5, 6.25])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([2.25, 6.5, 12.5, 18.5, 13.25])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([7.0, 11.75, 17.0, 12.5, 8.0])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([4.5, 10.75, 18.5, 24.5, 17.0])

    (indices, corr) = sc.statistics.xcorr(a, b, 2, sc.statistics.XCorrScale.Unbiased)
    assert indices.same_as([-2, -1, 0, 1, 2])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([8.5, 10.6667, 12.50, 12.6667, 12.50])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([4.5, 8.6667, 12.50, 24.6667, 26.5])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([14.0, 15.6667, 17.0, 16.6667, 16.0])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([9.0, 14.3333, 18.50, 32.6667, 34.0])

    (indices, corr) = sc.statistics.xcorr(a, b, 2, sc.statistics.XCorrScale.Coeff)
    assert indices.same_as([-2, -1, 0, 1, 2])
    # first column on a with first column in b
    assert corr[:, 0, 0].same_as([0.3347, 0.6300, 0.9844, 0.7481, 0.4922])
    # first column on a with second column in b
    assert corr[:, 0, 1].same_as([0.1180, 0.3408, 0.6554, 0.9700, 0.6947])
    # second column on a with first column on b
    assert corr[:, 1, 0].same_as([0.4109, 0.6897, 0.9978, 0.7337, 0.4696])
    # second column on a with second column on b
    assert corr[:, 1, 1].same_as([0.1759, 0.4201, 0.7230, 0.9575, 0.6644])


def test_stats_cross_covariance():
    a = sc.array([[1., 2, 3, 4], [2., 3, 4, 5]]).T
    b = sc.array([[3., 4, 5, 6], [7., 8, 9, 0]]).T

    # verified with octave
    # since xcov is just xcorr with means substracted, the test
    # simply checks for correct results for this scenario matching
    # octave results.
    (indices, cov) = sc.statistics.xcov(a, b)
    assert indices.same_as([-3, -2, -1, 0, 1, 2, 3.])
    # first column on a with first column in b
    assert cov[:, 0, 0].same_as([-2.25, -1.5, 1.25, 5.0, 1.25, -1.5, -2.25])
    # first column on a with second column in b
    assert cov[:, 0, 1].same_as([9, -1.5, -7.5, -10.0, 5, 3.5, 1.5])
    # second column on a with first column on b
    assert cov[:, 1, 0].same_as([-2.25, -1.5, 1.25, 5.0, 1.25, -1.5, -2.25])
    # second column on a with second column on b
    assert cov[:, 1, 1].same_as([9, -1.5, -7.5, -10.0, 5, 3.5, 1.5])


def test_stats_autocorrelation():
    a = sc.array([[1., 2, 3, 4], [2., 3, 4, 5]]).T
    # checked with octave
    results = sc.statistics.acorr(a)
    assert results[:, 0].same_as([4, 11, 20, 30, 20, 11, 4])
    assert results[:, 1].same_as([10, 23, 38, 54, 38, 23, 10])
    # print(sc.statistics.acorr(a, scale=XCorrScale.Coeff))


def test_stats_autocovariance():
    a = sc.array([[1., 2, 3, 4], [2., 3, 4, 5]]).T
    # checked with octave
    # the vectors are identical from a auto cov perspective
    results = sc.statistics.acov(a)
    assert results[:, 0].same_as([-2.25, -1.5, 1.25, 5.0, 1.25, -1.5, -2.25])
    assert results[:, 1].same_as([-2.25, -1.5, 1.25, 5.0, 1.25, -1.5, -2.25])

def test_stats_topk():
    test_data = sc.array([8, 3, 0, 7, 1, 5, 4, 6, 9, 2], dtype="float32")
    (vals, indices) = sc.statistics.topk_max(test_data, 5)
    assert vals.same_as([9,8,7,6,5])
    assert indices.same_as([8,0,3,7,5])
    (vals, indices) = sc.statistics.topk_min(test_data, 5)
    assert vals.same_as([0,1,2,3,4])
    assert indices.same_as([2,4,9,1,6])



