from __future__ import print_function
import numpy as np
from samplerate import resample, SampleRateConverter, resampling_callback

x = np.linspace(0, 10, 1000)
a = np.transpose([np.sin(2*np.pi*x), np.cos(2*np.pi*x)])
ratio = 2.0
converter = 2
b1 = resample(a, 2.0, converter, verbose=True)
src = SampleRateConverter(converter, 2)
b2 = src.process(a, 2.0, verbose=True)

def producer():
    #yield a
    yield np.transpose([np.sin(2*np.pi*x), np.cos(2*np.pi*x)])
    while True:
        yield None

with resampling_callback(lambda: next(producer()), ratio, converter, 2) as cb:
    b3 = cb.read(int(ratio) * a.shape[0])

print(b3.shape)
