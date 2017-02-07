"""Converters

"""
from __future__ import print_function, division
import numpy as np


def resample(input_data, ratio, converter_type, verbose=False):
    from .lowlevel import src_simple
    from .exceptions import SampleRateError

    input_data = np.require(input_data, requirements='C', dtype=np.float32)
    if input_data.ndim == 2:
        num_frames, channels = input_data.shape
        output_shape = (int(num_frames * ratio), channels)
    elif input_data.ndim == 1:
        num_frames, channels = input_data.size, 1
        output_shape = (int(num_frames * ratio), )
    else:
        raise ValueError('rank > 2 not supported')

    output_data = np.empty(output_shape, dtype=np.float32)

    (error, input_frames_used, output_frames_gen) \
        = src_simple(input_data, output_data, ratio, converter_type, channels)

    if error != 0:
        raise SampleRateError(error)

    if verbose:
        info = ('samplerate info:\n'
                '{} input frames used\n'
                '{} output frames generated\n'
                .format(input_frames_used, output_frames_gen))
        print(info)

    return (output_data[:output_frames_gen, :] if channels > 1 else
            output_data[:output_frames_gen])


class SampleRateConverter(object):

    def __init__(self, converter_type, channels):
        from .lowlevel import ffi, src_new, src_delete
        from .exceptions import SampleRateError

        state, error = src_new(converter_type, channels)
        self._state = ffi.gc(state, src_delete)
        self._converter_type = converter_type
        self._channels = channels
        if error != 0:
            raise SampleRateError(error)

    @property
    def converter_type(self):
        return self._converter_type

    @property
    def channels(self):
        return self._channels

    def reset(self):
        from .lowlevel import src_reset
        return src_reset(self._state)

    def set_ratio(self, new_ratio):
        from .lowlevel import src_set_ratio
        return src_set_ratio(self._state, new_ratio)

    def process(self, input_data, ratio, end_of_input=0, verbose=False):
        from .lowlevel import src_process
        from .exceptions import SampleRateError

        input_data = np.require(input_data, requirements='C', dtype=np.float32)
        if input_data.ndim == 2:
            num_frames, channels = input_data.shape
            output_shape = (int(num_frames * ratio), channels)
        elif input_data.ndim == 1:
            num_frames, channels = input_data.size, 1
            output_shape = (int(num_frames * ratio), )
        else:
            raise ValueError('rank > 2 not supported')

        if channels != self._channels:
            raise ValueError('Invalid number of channels in input data.')

        output_data = np.empty(output_shape, dtype=np.float32)

        (error,
         input_frames_used,
         output_frames_gen) = src_process(
             self._state, input_data, output_data, ratio, end_of_input)

        if error != 0:
            raise SampleRateError(error)

        if verbose:
            info = ('samplerate info:\n'
                    '{} input frames used\n'
                    '{} output frames generated\n'
                    .format(input_frames_used, output_frames_gen))
            print(info)

        return (output_data[:output_frames_gen, :] if channels > 1 else
                output_data[:output_frames_gen])


class CallbackResampler(object):

    def __init__(self, callback, ratio, converter_type, channels):
        if channels < 1:
            raise ValueError('Invalid number of channels.')
        self._callback = callback
        self._ratio = ratio
        self._converter_type = converter_type
        self._channels = channels
        self._state = None
        self._handle = None

    def create(self):
        from .lowlevel import ffi, src_callback_new, src_delete
        from .exceptions import SampleRateError

        state, handle, error = src_callback_new(
            self._callback, self._converter_type, self._channels
        )
        if error != 0:
            raise SampleRateError(error)
        self._state = ffi.gc(state, src_delete)
        self._handle = handle

    def destroy(self):
        if self._state:
            self._state = None
            self._handle = None

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.destroy()

    def set_starting_ratio(self, ratio):
        """ Set the starting conversion ratio for the next `read` call. """
        from .lowlevel import src_set_ratio
        src_set_ratio(self._state, ratio)
        self.ratio = ratio

    def reset(self):
        from .lowlevel import src_reset
        src_reset(self._state)

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self._ratio = ratio

    def read(self, num_frames):
        from .lowlevel import src_callback_read, src_error
        from .exceptions import SampleRateError

        if self._state is None:
            self.create()
        if self._channels > 1:
            output_shape = (num_frames, self._channels)
        elif self._channels == 1:
            output_shape = (num_frames, )
        output_data = np.empty(output_shape, dtype=np.float32)

        ret = src_callback_read(self._state, self._ratio, num_frames,
                                output_data)
        if ret == 0:
            error = src_error(self._state)
            if error:
                raise SampleRateError(error)

        return (output_data[:ret, :] if self._channels > 1 else
                output_data[:ret])


def resampling_callback(*args):
    return CallbackResampler(*args)
