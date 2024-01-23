import numpy as np
import samplerate


def test_resize():
    np.random.seed(0)
    ratio = 0.9
    x = np.random.randn(167)
    # internally, the bindings will first prepare a buffer of size
    # ceil(167 * 0.9) = 151, which will be resized to 150
    y = samplerate.resample(x, 0.9)
    assert y.shape[0] == 150
