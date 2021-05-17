from __future__ import annotations

from ..compute import ShapeletsArray, array as scarray
import numpy as np
import pathlib
import os

def load_dataset(name:str, dtype: np.dtype) -> ShapeletsArray:
    current_path = pathlib.Path(__file__).parent.absolute()
    file = os.path.join(current_path, name)
    data = np.loadtxt(file)
    return scarray(data, dtype=dtype)

def load_mat(name: str, section: str = "data") -> ShapeletsArray:
    current_path = pathlib.Path(__file__).parent.absolute()
    file = os.path.join(current_path, name)
    import scipy.io
    mat = scipy.io.loadmat(file)
    return scarray(mat[section])

__all__ = [ "load_dataset", "load_mat"]
