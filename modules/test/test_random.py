import shapelets.compute as sc

def test_random_sanity_check():
    rng = sc.random.random_engine()
    rng.beta(1.0, 1.0)
    rng.chisquare(3.0)
    rng.exponential(1.0)
    rng.gamma(1.0, 1.0)
    rng.logistic(1.0, 1.0)
    rng.lognormal(1.0, 1.0)
    rng.normal()
    rng.standard_normal()
    rng.uniform()
    rng.wald(1.0, 1.0)
    rng.multivariate_normal([1.0, 2.0], [[2.0, 0.3], [0.3, 4.0]], 10)
    

def test_random_gamma():
    sc.set_backend('opencl')
    rng = sc.random.random_engine('default', seed=1234)
    assert rng.gamma(5, 3, (3, 2)).same_as([[5.9197, 18.5246], [18.5167, 18.6795], [12.9743, 4.4505]])
