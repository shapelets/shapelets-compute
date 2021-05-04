import shapelets.compute as sc

def test_matprof_iface():
    tss = sc.cumsum(sc.random.randn((100, 3)), 0)
    r = sc.matrixprofile.matrix_profile(tss, 10)
    assert r.window == 10
    assert r.index.shape == (91, 3)
    assert r.profile.shape == r.index.shape
