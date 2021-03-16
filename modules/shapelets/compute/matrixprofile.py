
from ._pygauss import (
    matrixprofile as _matrixprofile, 
    mass as _mass, 
    matrixprofileLR as _matrixprofileLR)

## Matrix Profile
# def matrixprofile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> Tuple[ShapeletsArray, ShapeletsArray]:...
# def matrixprofileLR(ta: ArrayLike, m: int) -> Dict[str, Tuple[ShapeletsArray, ShapeletsArray]]:...
# def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray: ...


def matrix_profile(ta, m, tb = None): 
    return _matrixprofile(ta, m, tb)

def mass(queries, series):
    return _mass(queries, series)


