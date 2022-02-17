# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in  LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import pytest
import shapelets_compute.compute as sc


# def test_random_sanity_check():
#     rng = sc.random.random_engine()
#     rng.beta(1.0, 1.0)
#     rng.chisquare(3.0)
#     rng.exponential(1.0)
#     rng.gamma(5.0, 3.0)
#     rng.logistic(1.0, 1.0)
#     rng.lognormal(1.0, 1.0)
#     rng.normal()
#     rng.standard_normal()
#     rng.uniform()
#     rng.wald(1.0, 1.0)
#     rng.multivariate_normal([1.0, 2.0], [[2.0, 0.3], [0.3, 4.0]], 10)


@pytest.mark.skip('Flaky test on Windows 10 MSI environment')
def test_random_gamma():
    if 'opencl' in sc.get_available_backends():
        sc.set_backend('opencl')
        rng = sc.random.random_engine('default', seed=1234)
        assert rng.gamma(5, 3, (3, 2)).same_as([[5.9197, 18.5246], [18.5167, 18.6795], [12.9743, 4.4505]])
    assert True
