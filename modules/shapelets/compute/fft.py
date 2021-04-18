from __future__ import annotations

from typing import Literal, Optional, Sequence
import collections
from ._array_obj import array as as_array, ShapeletsArray
from .__basic_typing import ArrayLike

from ._pygauss import (
    _fft, _rfft,
    _ifft, _irfft,
    fftfreq, rfftfreq,
    fftNorm,
    spectral_derivative,
    fftshift
)

NormType = Literal["backward", "ortho", "forward"]

def _convertNorm(norm=None):
    if norm is None or (norm == "backward"):
        return fftNorm.Backward
    elif (norm == "ortho"):
        return fftNorm.Ortho
    elif (norm == "forward"):
        return fftNorm.Forward
    else:
        return float(norm)


def fft(a: ArrayLike, n: Optional[int] = None, axis: Optional[int] = -1, norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != -1 and axis != 0:
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0")

    sha = as_array(a)
    shape = sha.shape
    if n is not None:
        if isinstance(n, int):
            shape = (n, 1, 1, 1)
        else:
            ValueError("n should be an integer or None")

    return _fft(sha, _convertNorm(norm), shape)


def ifft(a: ArrayLike, n: Optional[int] = None, axis: Optional[int] = -1, norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != -1 and axis != 0:
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0")

    sha = as_array(a)
    shape = sha.shape
    if n is not None:
        if isinstance(n, int):
            shape = (n, 1, 1, 1)
        else:
            ValueError("n should be an integer or None")

    return _ifft(sha, _convertNorm(norm), shape)


def fft2(a: ArrayLike, n: Optional[Sequence[int]] = None, axis: Optional[Sequence[int]] = (-2, -1), norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != (-2, -1) and axis != (0, 1):
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0 and 1")

    sha = as_array(a)
    shape = sha.shape

    if n is not None:
        if isinstance(n, collections.Sequence) and len(n) == 2:
            shape[0] = n[0]
            shape[1] = n[1]
        else:
            raise ValueError("n should be a Sequence of two entries")

    return _fft(sha, _convertNorm(norm), shape)


def ifft2(a: ArrayLike, n: Optional[Sequence[int]] = None, axis: Optional[Sequence[int]] = (-2, -1), norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != (-2, -1) and axis != (0, 1):
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0 and 1")

    sha = as_array(a)
    shape = sha.shape

    if n is not None:
        if isinstance(n, collections.Sequence) and len(n) == 2:
            shape[0] = n[0]
            shape[1] = n[1]
        else:
            raise ValueError("n should be a Sequence of two entries")

    return _ifft(sha, _convertNorm(norm), shape)


def fftn(a: ArrayLike, n: Optional[Sequence[int]] = None, axis: Optional[Sequence[int]] = None, norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != (-3, -2, -1) and axis != (0, 1, 2):
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0, 1 and 2")

    sha = as_array(a)
    shape = sha.shape

    if n is not None:
        if isinstance(n, collections.Sequence):
            if len(n) > 3:
                raise ValueError(
                    "Shapelets supports a maximum of three axes in fft operations.")

            for i in range(len(n)):
                shape[i] = n[i]
        else:
            raise ValueError("n should be a Sequence of two entries")

    return _fft(sha, _convertNorm(norm), shape)


def ifftn(a: ArrayLike, n: Optional[Sequence[int]] = None, axis: Optional[Sequence[int]] = None, norm: Optional[NormType] = None) -> ShapeletsArray:
    if axis is not None and axis != (-3, -2, -1) and axis != (0, 1, 2):
        raise ValueError(
            "Shapelets doesn't support arbitrary axis.  The transformation always occurs on dimension 0, 1 and 2")

    sha = as_array(a)
    shape = sha.shape

    if n is not None:
        if isinstance(n, collections.Sequence):
            if len(n) > 3:
                raise ValueError(
                    "Shapelets supports a maximum of three axes in fft operations.")

            for i in range(len(n)):
                shape[i] = n[i]
        else:
            raise ValueError("n should be a Sequence of two entries")

    return _ifft(sha, _convertNorm(norm), shape)


__all__ = [
    "fftfreq",
    "rfftfreq",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "fftshift",
    "spectral_derivative",
    "NormType"
]
