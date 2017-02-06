""" Python bindings for libsamplerate based on CFFI and NumPy.
"""
from __future__ import print_function, division
import os
import sys
from ctypes.util import find_library

import numpy as np

from _samplerate import ffi

__version__ = '0.0.1'

# pylint: disable=invalid-name
_lib_basename = 'libsamplerate'
_lib_filename = find_library('samplerate')
if _lib_filename is None:
    if sys.platform == 'darwin':
        _lib_filename = '{}.dylib'.format(_lib_basename)
    elif sys.platform == 'win32':
        from platform import architecture
        _lib_filename = '{}-{}.dll'.format(_lib_basename, architecture()[0])
    else:
        raise OSError('{} not found'.format(_lib_basename))
    _lib_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '_samplerate_data', _lib_filename
    )

lib = ffi.dlopen(_lib_filename)


(SRC_SINC_BEST_QUALITY, SRC_SINC_MEDIUM_QUALITY,
 SRC_SINC_FASTEST, SRC_ZERO_ORDER_HOLD,
 SRC_LINEAR) = range(5)


def _check_data(data):
    """ Check whether `data` is a valid input/output for libsamplerate. """
    if not (data.dtype == np.float32 and data.flags.c_contiguous):
        raise ValueError('supplied data must be float32 and C contiguous')
    if data.ndim == 2:
        num_frames, channels = data.shape
    elif data.ndim == 1:
        num_frames, channels = data.size, 1
    else:
        raise ValueError('rank > 2 not supported')
    return num_frames, channels


def _src_strerror(error):
    return ffi.string(lib.src_strerror(error)).decode()


def _src_get_name(converter_type):
    return ffi.string(lib.src_get_name(converter_type)).decode()


def _src_get_description(converter_type):
    return ffi.string(lib.src_get_description(converter_type)).decode()


def _src_get_version():
    return ffi.string(lib.src_get_version()).decode()


def _src_simple(input_data, output_data, ratio, converter_type, channels):
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    error = lib.src_simple(data, converter_type, channels)
    return error, data.input_frames_used, data.output_frames_gen


def _src_new(converter_type, channels):
    error = ffi.new('int*')
    state = lib.src_new(converter_type, channels, error)
    return state, error[0]


def _src_delete(state):
    lib.src_delete(state)


def _src_process(state, input_data, output_data, ratio, end_of_input=0):
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    data.end_of_input = end_of_input
    error = lib.src_process(state, data)
    return error, data.input_frames_used, data.output_frames_gen


def _src_error(state):
    return lib.src_error(state) if state else None


def _src_reset(state):
    return lib.src_reset(state) if state else None


def _src_set_ratio(state, new_ratio):
    return lib.src_set_ratio(state, new_ratio) if state else None


def _src_is_valid_ratio(ratio):
    return bool(lib.src_is_valid_ratio(ratio))


@ffi.callback('src_callback_t')
def _src_input_callback(cb_data, data):
    cb_data = ffi.from_handle(cb_data)
    ret = cb_data['callback']()
    if ret is None:
        cb_data['last_input'] = None
        return 0  # No frames supplied
    input_data = np.require(ret, requirements='C', dtype=np.float32)
    input_frames, channels = _check_data(input_data)

    # Check whether the correct number of channels is supplied by user.
    if cb_data['channels'] != channels:
        raise ValueError('Invalid number of channels in callback.')

    # Store a reference of the input data to ensure it is still alive when
    # accessed by libsamplerate.
    cb_data['last_input'] = input_data

    data[0] = ffi.cast('float*', ffi.from_buffer(input_data))
    return input_frames


def _src_callback_new(callback, converter_type, channels):
    cb_data = {'callback': callback, 'channels': channels}
    handle = ffi.new_handle(cb_data)
    error = ffi.new('int*')
    state = lib.src_callback_new(_src_input_callback, converter_type,
                                 channels, error, handle)
    if state == ffi.NULL:
        return None, handle, error[0]
    return state, handle, error[0]


def _src_callback_read(state, ratio, frames, data):
    data_ptr = ffi.cast('float*', ffi.from_buffer(data))
    return lib.src_callback_read(state, ratio, frames, data_ptr)


__libsamplerate_version__ = _src_get_version()
if __libsamplerate_version__.startswith(_lib_basename):
    __libsamplerate_version__ = __libsamplerate_version__[
        len(_lib_basename) + 1:__libsamplerate_version__.find(' ')
    ]


class SampleRateError(RuntimeError):

    def __init__(self, error):
        message = 'libsamplerate error #{}: {}'.format(
            error, _src_strerror(error)
        )
        super(SampleRateError, self).__init__(message)
        self.error = error


def resample(input_data, ratio, converter_type, verbose=False):
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
        = _src_simple(input_data, output_data, ratio, converter_type, channels)

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
        state, error = _src_new(converter_type, channels)
        self._state = ffi.gc(state, _src_delete)
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
        return _src_reset(self._state)

    def set_ratio(self, new_ratio):
        return _src_set_ratio(self._state, new_ratio)

    def process(self, input_data, ratio, end_of_input=0, verbose=False):
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
         output_frames_gen) = _src_process(
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
        state, handle, error = _src_callback_new(
            self._callback, self._converter_type, self._channels
        )
        if error != 0:
            raise SampleRateError(error)
        self._state = ffi.gc(state, _src_delete)
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
        _src_set_ratio(self._state, ratio)
        self.ratio = ratio

    def reset(self):
        _src_reset(self._state)

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self._ratio = ratio

    def read(self, num_frames):
        if self._state is None:
            self.create()
        if self._channels > 1:
            output_shape = (num_frames, self._channels)
        elif self._channels == 1:
            output_shape = (num_frames, )
        output_data = np.empty(output_shape, dtype=np.float32)

        ret = _src_callback_read(self._state, self._ratio, num_frames,
                                 output_data)
        if ret == 0:
            error = _src_error(self._state)
            if error:
                raise SampleRateError(error)

        return (output_data[:ret, :] if self._channels > 1 else
                output_data[:ret])


def resampling_callback(*args):
    return CallbackResampler(*args)
