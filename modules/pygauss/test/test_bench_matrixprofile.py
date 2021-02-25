import pygauss as sh
import matrixprofile as mp
import stumpy as st
import numpy as np


def st_stump(data):
    return st.stump(data, 100)


def mp_org_stomp(data, jobs=1):
    return mp.algorithms.stomp(data, 100, jobs)


def mp_org_mpx(data, jobs=1):
    return mp.algorithms.mpx(data, 100, jobs)


def sh_matrixprofile(data):
    return sh.matrixprofile(data, 100)


def test_st_stump(benchmark):
    # ds = mp.datasets.load('motifs-discords-small')
    ds = mp.datasets.load('8192')
    data = ds['data']
    benchmark(st_stump, data)


def test_mp_org_stomp(benchmark):
    # ds = mp.datasets.load('motifs-discords-small')
    ds = mp.datasets.load('8192')
    data = ds['data']
    benchmark(mp_org_stomp, data)


def test_mp_org_mpx(benchmark):
    # ds = mp.datasets.load('motifs-discords-small')
    ds = mp.datasets.load('8192')
    data = ds['data']
    benchmark(mp_org_mpx, data)


def test_sh_cpu(benchmark):
    sh.set_backend(sh.Backend.CPU)
    ds = mp.datasets.load('8192')
    # ds = mp.datasets.load('motifs-discords-small')
    data = sh.array(ds['data'])
    benchmark(sh_matrixprofile, data)


