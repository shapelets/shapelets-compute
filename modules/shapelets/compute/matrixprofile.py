<<<<<<< HEAD
from typing import Optional, Tuple
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray
=======
>>>>>>> master

from ._pygauss import (
    matrixprofile as _matrixprofile, 
    mass as _mass, 
<<<<<<< HEAD
    matrixprofileLR as _matrixprofileLR,
    mpdist_vect as _mpdist_vect,
    snippets, Snippet
    )
=======
    matrixprofileLR as _matrixprofileLR)
>>>>>>> master

## Matrix Profile
# def matrixprofile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> Tuple[ShapeletsArray, ShapeletsArray]:...
# def matrixprofileLR(ta: ArrayLike, m: int) -> Dict[str, Tuple[ShapeletsArray, ShapeletsArray]]:...
# def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray: ...


<<<<<<< HEAD
def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray:
    return _mass(queries, series)

def matrix_profile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> Tuple[ShapeletsArray, ShapeletsArray]:
    return _matrixprofile(ta, m, tb)

def mpdist_vect(ts: ArrayLike, tsb: ArrayLike, w: int, threshold: Optional[float] = 0.05) -> ShapeletsArray:
    return _mpdist_vect(ts, tsb, w, threshold)
=======
def matrix_profile(ta, m, tb = None): 
    return _matrixprofile(ta, m, tb)

def mass(queries, series):
    return _mass(queries, series)

>>>>>>> master

