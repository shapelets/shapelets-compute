import numpy as np
import shapelets.compute as sh 

def test_1d_cases():
    a = np.array([0.,1,2,3,4,3,2,1,0,1], dtype="float32")
    assert sh.fft.fft(a).same_as(np.fft.fft(a))
    assert sh.fft.fft(a, norm="forward").same_as(np.fft.fft(a, norm="forward"))
    assert sh.fft.fft(a, norm="ortho").same_as(np.fft.fft(a, norm="ortho"))
    assert sh.fft.fft(a, norm="backward").same_as(np.fft.fft(a, norm="backward"))
    assert sh.fft.fft(a, shape=5).same_as(np.fft.fft(a,n=5))
    assert sh.fft.fft(a, shape=15).same_as(np.fft.fft(a,n=15))
    
    b = np.fft.fft(a)
    assert sh.fft.ifft(b).same_as(np.fft.ifft(b))
    assert sh.fft.ifft(b, norm="forward").same_as(np.fft.ifft(b, norm="forward"))
    assert sh.fft.ifft(b, norm="ortho").same_as(np.fft.ifft(b, norm="ortho"))
    assert sh.fft.ifft(b, norm="backward").same_as(np.fft.ifft(b, norm="backward"))

    b = np.fft.fft(a, n=5)
    assert sh.fft.ifft(b).same_as(np.fft.ifft(b))

    b = np.fft.fft(a, n=50)
    assert sh.fft.ifft(b).same_as(np.fft.ifft(b))

    npa = np.array([0,1,2,3,4,3,2,1,0,1,2,3], dtype="float32").reshape(4,3)
    sha = sh.array(npa)
    assert sh.fft.fft(sha[::,0]).same_as(np.fft.fft(npa[::,0]))
    assert sh.fft.fft(sha[::,1]).same_as(np.fft.fft(npa[::,1]))
    assert sh.fft.fft(sha[::,2]).same_as(np.fft.fft(npa[::,2]))
    assert sh.fft.fft(sha[0,::]).same_as(np.fft.fft(npa[0,::]))
    assert sh.fft.fft(sha[1,::]).same_as(np.fft.fft(npa[1,::]))
    assert sh.fft.fft(sha[2,::]).same_as(np.fft.fft(npa[2,::]))
    assert sh.fft.fft(sha[3,::]).same_as(np.fft.fft(npa[3,::]))
