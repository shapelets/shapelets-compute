from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Union

import shapelets.compute as sc

import math
import random

class DeltaGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_delta(self, n: int, rng) -> sc.ShapeletsArray: ...

    def __add__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: self.generate_delta(n, rng) + other.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: self.generate_delta(n, rng) + other)
        raise ValueError("Unsupported composition with " + type(other))

    def __mul__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: self.generate_delta(n, rng) * other.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: self.generate_delta(n, rng) * other)
        raise ValueError("Unsupported composition with " + type(other))

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: self.generate_delta(n, rng) - other.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: self.generate_delta(n, rng) - other)
        raise ValueError("Unsupported composition with " + type(other))

    def __rsub__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: other.generate_delta(n, rng) - self.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: other - self.generate_delta(n, rng))
        raise ValueError("Unsupported composition with " + type(other))

    def __truediv__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: self.generate_delta(n, rng) / other.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: self.generate_delta(n, rng) / other)
        raise ValueError("Unsupported composition with " + type(other))

    def __rtruediv__(self, other):
        if isinstance(other, DeltaGenerator):
            return _Derived(lambda n, rng: other.generate_delta(n, rng) / self.generate_delta(n, rng))
        elif type(other) == int or type(other) == float:
            return _Derived(lambda n, rng: other / self.generate_delta(n, rng))
        raise ValueError("Unsupported composition with " + type(other))


class _Derived(DeltaGenerator):
    def __init__(self, fn: Callable[[int, sc.random.ShapeletsRandomEngine], DeltaGenerator]) -> None:
        self.fn = fn

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        return self.fn(n, rng)


class _Cyclic(DeltaGenerator):
    def __init__(self, amp: Tuple[int, int], freq: Tuple[int, int]):
        self.amp = amp
        self.freq = freq

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        a = sc.tile(rng.uniform(*self.amp, 1), n+1)
        f = sc.tile(rng.uniform(*self.freq, 1), n+1)
        t = sc.iota(n+1, dtype=f.dtype)
        values = a*sc.sin((2.0 * math.pi * t) / f)
        return sc.diff1(values, 0)


class _Normal(DeltaGenerator):
    def __init__(self):
        """ does nothing """
        pass

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        return sc.zeros(n)

class _IncDec(DeltaGenerator):
    def __init__(self, increasing: bool, grad: Tuple[int, int]):
        self.grad = grad
        self.increasing = increasing

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        g = sc.tile(rng.uniform(*self.grad), n)
        if not self.increasing:
            return -g
        return g


class _UpDown(DeltaGenerator):
    def __init__(self, upwards: bool, bump: Tuple[int, int] = (7.5, 20)):
        self.bump = bump
        self.upwards = upwards

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        x = rng.uniform(*self.bump)
        if not self.upwards:
            x = -x

        t3 = rng.uniform(int(n/3), int(2*n/3))
        t = sc.iota(n+1)
        values = sc.zeros(n+1)
        values[t >= sc.tile(t3, n+1)] = x
        return sc.diff1(values, 0)


class _WhiteNoise(DeltaGenerator):
    def __init__(self, b0: float, fs: float):
        self.adj = math.sqrt(b0*fs/2.0)

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        values = self.adj * rng.standard_normal(n+1)
        return sc.diff1(values, 0)


class _BrownNoise(DeltaGenerator):
    def __init__(self,  b_minus2: float, fs: float):
        self.scale = 1.0 / fs
        b0 = b_minus2*(4.0*math.pi*math.pi)
        self.adj = math.sqrt(b0*fs/2.0)

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        values = self.adj * rng.standard_normal(n+1)
        brown = self.scale * sc.cumsum(values)
        return sc.diff1(brown, 0)


class _VioletNoise(DeltaGenerator):
    def __init__(self, b2: float, fs: float):
        b0 = b2/(2.0*math.pi)**2
        self.fs = fs
        self.adj = math.sqrt(b0*fs/2.0)

    def generate_delta(self, n: int, rng: sc.random.ShapeletsRandomEngine) -> sc.ShapeletsArray:
        white = self.adj*rng.standard_normal(n+2)
        violet = self.fs * sc.diff1(white, 0)
        return sc.diff1(violet, 0)


def cc_downward(bump: Tuple[int, int] = (7.5, 20)) -> DeltaGenerator:
    """
    Creates a downward control chart sequence

    Parameters
    ----------
    bump: Tuple. Defaults to (7.5, 20)
          Randomly chooses the size of the step
    """
    return _UpDown(False, bump)


def cc_upward(bump: Tuple[int, int] = (7.5, 20)) -> DeltaGenerator:
    """
    Creates an upward control chart sequence

    Parameters
    ----------
    bump: Tuple. Defaults to (7.5, 20)
          Randomly chooses the size of the step
    """
    return _UpDown(True, bump)


def cc_decreasing(grad: Tuple[int, int] = (0.2, 0.5)) -> DeltaGenerator:
    """
    Creates an ever decreasing control chart sequence

    Parameters
    ----------
    grad: Tuple.  Defaults to (0.2, 0.5)
          Chooses randomly a gradient between the two numbers
    """
    return _IncDec(False, grad)


def cc_increasing(grad: Tuple[int, int] = (0.2, 0.5)) -> DeltaGenerator:
    """
    Creates an ever increasing control chart sequence

    Parameters
    ----------
    grad: Tuple.  Defaults to (0.2, 0.5)
          Chooses randomly a gradient between the two numbers
    """
    return _IncDec(True, grad)


def cc_normal() -> DeltaGenerator:
    """
    Creates a normal shape control chart sequence
    """
    return _Normal()


def cc_cyclic(amp: Tuple[int, int] = (10, 15), freq: Tuple[int, int] = (10, 15)) -> DeltaGenerator:
    """
    Creates a cyclic shape control chart sequence

    Parameters
    ----------
    amp:  Tuple.  Defaults to (10,15)
          Chooses randomly a number between the tuple values as amplitude of the oscillation.
    freq: Tuple. Defaults to (10.15)
          Chooses the frequency of the oscillation randomly between the values of the tuple.
    """
    return _Cyclic(amp, freq)


def white_noise(b0: float = 1.0, fs: float = 1.0) -> DeltaGenerator:
    """ White noise generator

        Generate time series with white noise that has constant Power Spectral Density = :math:`b_0`,
        up to the nyquist frequency :math:`\\frac{f_s}{2.0}`.

        The Power Spectral Density is at 'height' :math:b_0` and extends from 0 Hz up to the nyquist
        frequency :math:`\\frac{f_s}{2.0}` (prefactor :math:`\\sqrt{\\frac{b_0*f_s}{2.0}}`)

        Parameters
        ----------
        b0: float, optional (default 1.0)
            desired power-spectral density in [X^2/Hz] where X is the unit of x
        fs: float, option (default 1.0)
            sampling frequency, i.e. 1/fs is the time-interval between datapoints
    """
    return _WhiteNoise(b0, fs)


def brown_noise(b_minus2: float = 1.0, fs: float = 1.0) -> DeltaGenerator:
    """ Brownian or random walk (diffusion) noise with 1/f^2 PSD

    Not really a color... rather Brownian or random-walk.
    Obtained by integrating white-noise.

    Parameters
    ----------
    b_minus2: float, optional (default 1.0)
        desired power-spectral density is b2*f^-2
    fs: float, optional (default 1.0)
        sampling frequency, i.e. 1/fs is the time-interval between datapoints
    """
    return _BrownNoise(b_minus2, fs)


def violet_noise(b2: float = 1.0, fs: float = 1.0) -> DeltaGenerator:
    """ Violet noise with f^2 PSD

    Obtained by differentiating white noise

    Parameters
    ----------
    b2: float, optional (default 1.0)
        desired power-spectral density is b2*f^2
    fs: float, optional (default 1.0)
        sampling frequency, i.e. 1/fs is the time-interval between datapoints
    """
    return _VioletNoise(b2, fs)

# missing pink

def _run_choice(py_rnd: random.Random, sh_rnd: sc.random.ShapeletsRandomEngine, n: int, choices: List, prob: List) -> sc.ShapeletsArray:
    gen = py_rnd.choices(choices, prob, k=1)[0]
    return gen.generate_delta(n, sh_rnd)

def _run_single(sh_rnd: sc.random.ShapeletsRandomEngine, n: int, delta: DeltaGenerator) -> sc.ShapeletsArray:
    return delta.generate_delta(n, sh_rnd)

def _process_single_program_instruction(x: DeltaGenerator, n: int, py_rnd: random.Random, sh_rnd: sc.random.ShapeletsRandomEngine):
    return (lambda z,e: lambda: _run_single(sh_rnd, z,e))(n, x)

def _process_multiple_program_instruction(x: List, n: int, py_rnd: random.Random, sh_rnd: sc.random.ShapeletsRandomEngine):
    l = len(x)
    prob = [1.0/l]*l 

    if isinstance(x[0], List):
        if l>1:
            prob = x[1]    
        return (lambda z,c,p: lambda: _run_choice(py_rnd, sh_rnd, z,c,p))(n, x[0], prob)
    
    return (lambda z,c,p: lambda: _run_choice(py_rnd, sh_rnd, z,c,p))(n, x, prob)

def _process_program_instruction(x: Union[List, DeltaGenerator], n: int, py_rnd: random.Random, sh_rnd: sc.random.ShapeletsRandomEngine):
    if isinstance(x, List) and len(x) > 0:
        return _process_multiple_program_instruction(x, n, py_rnd, sh_rnd)     
    elif isinstance(x, DeltaGenerator):
        return _process_single_program_instruction(x, n, py_rnd, sh_rnd)
    
    raise ValueError("Unknown element in list " + str(x)) 

def generate(program: List = [], lengths: Union[int, List[int]] = 10, start_level: float = 0.0, repetitions: int = 1): 
    if len(program) == 0 or (isinstance(lengths, List) and len(lengths) != len(program)):
        raise ValueError("No valid configuration provided")

    seed=random.randint(0, 1000000)
    sh_rnd = sc.random.random_engine(seed = seed)
    py_rnd = random.Random(seed)

    if isinstance(lengths, int):
        lengths = [lengths] * len(program)

    instructions = [_process_program_instruction(x, n, py_rnd, sh_rnd) for (x,n) in zip(program, lengths)]
    data =[i() for _ in range(repetitions) for i in instructions ]
    return sc.cumsum(sc.join(data, 0)) + start_level;


__all__ = [
    "generate",
    "violet_noise",
    "brown_noise",
    "white_noise",
    "cc_cyclic",
    "cc_normal",
    "cc_increasing",
    "cc_decreasing", 
    "cc_upward", 
    "cc_downward",
    "DeltaGenerator"
 ]


if __name__ == '__main__':

    import matplotlib.pyplot as plt 

    x = sc.random.random_engine()

    program = [
        # white_noise(),
        # brown_noise(),
        # violet_noise() + white_noise()
        # cc_increasing() + cc_decreasing() + cc_cyclic() + cc_normal() + cc_upward()+cc_downward(),
        [cc_upward(), cc_normal()],
        [[cc_cyclic(), cc_normal()], [.9, .1]],
        cc_decreasing() + cc_cyclic() + white_noise(),
        # cc_increasing(),
        # cc_decreasing(),
        # cc_downward() + 0.3*white_noise(),
        # cc_upward() + 0.3*white_noise(),
        
    ]

    # arr = sc.iota((5,3))
    # arr.display()
    # sc.random.permutation(arr, 0).display()
    # sc.random.permutation(arr, 1).display()
    # sc.random.permutation(10).display()

    # sh_rnd = sc.random.default_rng()
    # print(cc_increasing().generate_delta(10, sh_rnd).dtype)
    # print(cc_decreasing().generate_delta(10, sh_rnd).dtype)
    # print((cc_downward() + 0.3*white_noise()).generate_delta(10, sh_rnd).dtype)
    # print((cc_upward() + 0.3*white_noise()).generate_delta(10, sh_rnd).dtype)

    r = generate(program, 100, start_level=100.0, repetitions=1)
    plt.plot(r)
    plt.show()
    
