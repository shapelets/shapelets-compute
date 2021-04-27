from __future__ import annotations

from typing import Any, List, Optional, Tuple
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from ._pygauss import (
    matrixprofile as _matrixprofile, 
    mass as _mass, 
    matrixprofileLR as _matrixprofileLR,
    mpdist_vect as _mpdist_vect,
    Snippet, 
    )


## Matrix Profile
# def matrixprofile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> Tuple[ShapeletsArray, ShapeletsArray]:...
# def matrixprofileLR(ta: ArrayLike, m: int) -> Dict[str, Tuple[ShapeletsArray, ShapeletsArray]]:...
# def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray: ...


def mass(queries: ArrayLike, series: ArrayLike) -> ShapeletsArray:
    return _mass(queries, series)

def matrix_profile(ta: ArrayLike, m: int, tb: Optional[ArrayLike] = None) -> Tuple[ShapeletsArray, ShapeletsArray]:
    return _matrixprofile(ta, m, tb)

def mpdist_vect(ts: ArrayLike, tsb: ArrayLike, w: int, threshold: Optional[float] = 0.05) -> ShapeletsArray:
    return _mpdist_vect(ts, tsb, w, threshold)

def snippets(ts: ArrayLike, snippet_size: int, num_snippets: int, window_size: Optional[int] = None)-> List[Any]:
    import numpy as np 
    import matrixprofile as morg 
    return morg.discover.snippets(np.array(ts), snippet_size, num_snippets)
    
def cac(profile: ArrayLike, index: ArrayLike, window: int) -> Tuple[ShapeletsArray, int]:
    import numpy as np
    pos = np.arange(0, len(index))
    nnmark = np.zeros(len(index))
    small = np.minimum(pos, index)
    large = np.maximum(pos, index)
    
    for x in np.nditer(small): 
        nnmark[int(x)] = nnmark[int(x)] + 1

    for x in np.nditer(large):
        nnmark[int(x)] = nnmark[int(x)] - 1
        
    cac = np.cumsum(nnmark)
    l = len(cac)

    # iac is a parabolic curve of length l and height 1/2 n
    iac = np.fromfunction(lambda i:+0.00001 + (2*i*(l-i)/l), (l,))
    cac = np.minimum(cac / iac, 1.0)
    cac[0:window] = 1
    cac[-window:] = 1     
    return cac, np.argmin(cac)
