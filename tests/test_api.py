import numpy as np
import pytest

import samplerate


def test_aliases():
    from samplerate.converters import (
        Resampler,
        CallbackResampler,
        resample,
        ConverterType,
    )
    from samplerate import (
        Resampler,
        CallbackResampler,
        resample,
        ConverterType,
        ResamplingError,
    )
    from samplerate.exceptions import ResamplingError


@pytest.fixture(scope="module", params=[1, 2])
def data(request):
    num_channels = request.param
    periods = np.linspace(0, 10, 1000)
    input_data = [
        np.sin(2 * np.pi * periods + i * np.pi / 2) for i in range(num_channels)
    ]
    return (
        (num_channels, input_data[0])
        if num_channels == 1
        else (num_channels, np.transpose(input_data))
    )


@pytest.fixture(params=[0, 1, 2, 3, 4])
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
    _, input_data = data

    def producer():
        yield input_data
        while True:
            yield None

    callback = lambda p=producer(): next(p)
    channels = input_data.shape[-1] if input_data.ndim == 2 else 1

    resampler = samplerate.CallbackResampler(callback, ratio, converter_type, channels)
    resampler.read(int(ratio) * input_data.shape[0])


def test_callback_with(data, converter_type, ratio=2.0):
    from samplerate import CallbackResampler

    _, input_data = data

    def producer():
        yield input_data
        while True:
            yield None

    callback = lambda p=producer(): next(p)
    channels = input_data.shape[-1] if input_data.ndim == 2 else 1

    with CallbackResampler(
        callback, ratio, converter_type, channels=channels
    ) as resampler:
        resampler.read(int(ratio) * input_data.shape[0])


def test_callback_with_2x(data, converter_type, ratio=2.0):
    """
    Tests that there are no errors if we reuse an object created with a context manager
    """
    from samplerate import CallbackResampler

    _, input_data = data

    def producer():
        yield input_data
        while True:
            yield None

    channels = input_data.shape[-1] if input_data.ndim == 2 else 1

    callback = lambda p=producer(): next(p)

    with CallbackResampler(
        callback, ratio, converter_type, channels=channels
    ) as resampler:
        resampler.read(int(ratio) * input_data.shape[0] // 2)

    # re-initialize the data producer
    resampler.read(int(ratio) * input_data.shape[0] // 2)


def test_Resampler_clone():
    resampler = samplerate.Resampler("sinc_best", 1)
    new_resampler = resampler.clone()


def test_CallbackResampler_clone(data, converter_type, ratio=2.0):
    _, input_data = data

    def producer():
        yield input_data
        while True:
            yield None

    callback = lambda p=producer(): next(p)
    channels = input_data.shape[-1] if input_data.ndim == 2 else 1

    resampler = samplerate.CallbackResampler(callback, ratio, converter_type, channels)
    resampler.read(int(ratio) * input_data.shape[0])

    new_resampler = resampler.clone()


@pytest.mark.parametrize(
    "input_obj,expected_type",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        ("sinc_best", 0),
        ("sinc_medium", 1),
        ("sinc_fastest", 2),
        ("zero_order_hold", 3),
        ("linear", 4),
        (samplerate.ConverterType.sinc_best, 0),
        (samplerate.ConverterType.sinc_medium, 1),
        (samplerate.ConverterType.sinc_fastest, 2),
        (samplerate.ConverterType.zero_order_hold, 3),
        (samplerate.ConverterType.linear, 4),
    ],
)
def test_converter_type(input_obj, expected_type):
    ret = samplerate._internals.get_converter_type(input_obj)
    assert ret == expected_type
