import numpy as np
import pytest

import samplerate


@pytest.fixture(scope="module", params=[1, 2])
def data(request):
    num_channels = request.param
    periods = np.linspace(0, 10, 1000)
    input_data = [
        np.sin(2 * np.pi * periods + i * np.pi / 2)
        for i in range(num_channels)
    ]
    return ((num_channels, input_data[0])
            if num_channels == 1 else (num_channels, np.transpose(input_data)))


@pytest.fixture(params=list(samplerate.converters.ConverterType))
def converter_type(request):
    return request.param


def test_simple(data, converter_type, ratio=2.0):
    _, input_data = data
    samplerate.resample(input_data, ratio, converter_type)


def test_process(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    src = samplerate.Resampler(converter_type, num_channels)
    src.process(input_data, ratio)


def test_match(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    output_simple = samplerate.resample(input_data, ratio, converter_type)
    resampler = samplerate.Resampler(converter_type, channels=num_channels)
    output_full = resampler.process(input_data, ratio, end_of_input=True)
    assert np.allclose(output_simple, output_full)


def test_callback(data, converter_type, ratio=2.0):
    from samplerate import CallbackResampler
    _, input_data = data

    def producer():
        yield input_data
        while True:
            yield None

    callback = lambda p=producer(): next(p)

    with CallbackResampler(callback, ratio, converter_type) as resampler:
        resampler.read(int(ratio) * input_data.shape[0])
