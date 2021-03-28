import shapelets.compute as sc 
import numpy as np


def test_stats_skewness():
    rng = sc.random.default_rng()
    test_data = rng.exponential(shape=(10000000,1))
    result = sc.statistics.skewness(test_data)
    assert result.same_as([2.0], eps=0.01)
    assert result.shape == (1, 1)

def test_stats_kurtosis():
    rng = sc.random.default_rng()
    test_data = rng.exponential(shape=(10000000,1))
    result = sc.statistics.kurtosis(test_data)
    assert result.same_as([6.0], eps=0.01)
    assert result.shape == (1, 1)    

def test_stats_moment():
    rng = sc.random.default_rng()
    test_data = rng.exponential(shape=(10000000,1))
    # since scale is 1.0 -> lambda is also 1.0
    # then, exponential moments are n!, where 
    # n is the moment number
    m1 = sc.statistics.moment(test_data, 1)
    m2 = sc.statistics.moment(test_data, 2)
    m3 = sc.statistics.moment(test_data, 3)
    m4 = sc.statistics.moment(test_data, 4)
    assert m1.same_as(1.0, eps=0.1)
    assert m2.same_as(2.0, eps=0.1)
    assert m3.same_as(6.0, eps=0.1)
    assert m4.same_as(24.0, eps=0.1)


def test_stats_correlation():
    data = [[1., 2, 3, 4],
            [2., 3, 4, 5],
            [3., 4, 5, 6],
            [7., 8, 9, 0]]
    
    npr = np.corrcoef(data, rowvar=False)
    sr = sc.statistics.correlation(data, False)
    assert sr.same_as(npr)
    assert sc.corrcoef(data, rowvar=False).same_as(sr)
    assert sc.corrcoef(data).same_as(np.corrcoef(data))

def test_stats_covariance():
    data = [[1., 2, 3, 4],
            [2., 3, 4, 5],
            [3., 4, 5, 6],
            [7., 8, 9, 0]]

    npr = np.cov(data, rowvar = False)  
    sr = sc.statistics.covariance(data, True)
    assert sr.same_as(npr)                  
    assert sc.cov(data, rowvar=True, bias=False).same_as(np.cov(data, rowvar=True, bias=False))
    assert sc.cov(data, rowvar=True, bias=True).same_as(np.cov(data, rowvar=True, bias=True))
    assert sc.cov(data, rowvar=False, bias=False).same_as(np.cov(data, rowvar=False, bias=False))
    assert sc.cov(data, rowvar=False, bias=True).same_as(np.cov(data, rowvar=False, bias=True))    

def test_stats_cross_correlation():
    a1 = [1., 2, 3, 4]
    b1 = [3., 4, 5, 6]
    a = [[1., 2, 3, 4], [2., 3, 4, 5]]
    b = [[3., 4, 5, 6], [7., 8, 9, 0]]
    print(np.correlate(a1, b1, 'valid'))
    print(np.correlate(a1, b1, 'same'))
    print(np.correlate(a1, b1, 'full'))
    print(sc.statistics.cross_correlation(a, b, True))
    print(sc.statistics.cross_correlation(a, b, False))
    print(sc.statistics.cross_correlation(a1, b1, True))
    print(sc.statistics.cross_correlation(a1, b1, False)) 

def test_stats_cross_with_matlab():
    n = sc.arange(16)           # n = 0:15
    x = sc.power(0.84, n)       # x = 0.84.^n
    y = sc.shift(x, 5)          # y = circshift(x, 5)
    import matplotlib.pyplot as plt 
    plt.stem(sc.statistics.cross_correlation(x, y, False))
    plt.show()

def test_from_khiva():
    data1 = sc.array([[0., 1, 2, 3], [0, 1, 2, 3]]).T
    data2 = sc.array([[4., 6, 8, 10, 12],[4, 6, 8, 10, 12]]).T
    data1.display()
    data2.display()
    sc.statistics.cross_correlation(data1, data2, False).display()
    print(np.correlate(data1[:,0], data2[:,0], 'full'))
