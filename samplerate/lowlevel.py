"""Lowlevel wrappers around libsamplerate.

The docstrings of the `src_*` functions are adapted from the libsamplerate
header file.
"""
import os as _os
import sys as _sys
from ctypes.util import find_library as _find_library

import numpy as _np

# Locate and load libsamplerate
from samplerate._src import ffi
lib_basename = 'libsamplerate'
lib_filename = _find_library('samplerate')
if lib_filename is None:
    if _sys.platform == 'darwin':
        lib_filename = '{}.dylib'.format(lib_basename)
    elif _sys.platform == 'win32':
        from platform import architecture
        lib_filename = '{}-{}.dll'.format(lib_basename, architecture()[0])
    else:
        raise OSError('{} not found'.format(lib_basename))
    lib_filename = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), '_samplerate_data',
        lib_filename)

_lib = ffi.dlopen(lib_filename)


def _check_data(data):
    """Check whether `data` is a valid input/output for libsamplerate.

    Returns
    -------
    num_frames
        Number of frames in `data`.
    channels
        Number of channels in `data`.

    Raises
    ------
        ValueError: If invalid data is supplied.
    """
    if not (data.dtype == _np.float32 and data.flags.c_contiguous):
        raise ValueError('supplied data must be float32 and C contiguous')
    if data.ndim == 2:
        num_frames, channels = data.shape
    elif data.ndim == 1:
        num_frames, channels = data.size, 1
    else:
        raise ValueError('rank > 2 not supported')
    return num_frames, channels


def src_strerror(error):
    """Convert the error number into a string."""
    return ffi.string(_lib.src_strerror(error)).decode()


def src_get_name(converter_type):
    """Return the name of the converter given by `converter_type`."""
    return ffi.string(_lib.src_get_name(converter_type)).decode()


def src_get_description(converter_type):
    """Return the description of the converter given by `converter_type`."""
    return ffi.string(_lib.src_get_description(converter_type)).decode()


def src_get_version():
    """Return the version string of libsamplerate."""
    return ffi.string(_lib.src_get_version()).decode()


def src_simple(input_data, output_data, ratio, converter_type, channels):
    """Perform a single conversion from an input buffer to an output buffer.

    Simple interface for performing a single conversion from input buffer to
    output buffer at a fixed conversion ratio. Simple interface does not require
    initialisation as it can only operate on a single buffer worth of audio.
    """
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    error = _lib.src_simple(data, converter_type, channels)
    return error, data.input_frames_used, data.output_frames_gen


def src_new(converter_type, channels):
    """Initialise a new sample rate converter.

    Parameters
    ----------
    converter_type : int
        Converter to be used.
    channels : int
        Number of channels.

    Returns
    -------
    state
        An anonymous pointer to the internal state of the converter.
    error : int
        Error code.
    """
    error = ffi.new('int*')
    state = _lib.src_new(converter_type, channels, error)
    return state, error[0]


def src_delete(state):
    """Release `state`.

    Cleanup all internal allocations.
    """
    _lib.src_delete(state)


def src_process(state, input_data, output_data, ratio, end_of_input=0):
    """Standard processing function.

    Returns non zero on error.
    """
    input_frames, _ = _check_data(input_data)
    output_frames, _ = _check_data(output_data)
    data = ffi.new('SRC_DATA*')
    data.input_frames = input_frames
    data.output_frames = output_frames
    data.src_ratio = ratio
    data.data_in = ffi.cast('float*', ffi.from_buffer(input_data))
    data.data_out = ffi.cast('float*', ffi.from_buffer(output_data))
    data.end_of_input = end_of_input
    error = _lib.src_process(state, data)
    return error, data.input_frames_used, data.output_frames_gen


def src_error(state):
    """Return an error number."""
    return _lib.src_error(state) if state else None


def src_reset(state):
    """Reset the internal SRC state.

    Does not modify the quality settings.
    Does not free any memory allocations.
    Returns non zero on error.
    """
    return _lib.src_reset(state) if state else None


def src_set_ratio(state, new_ratio):
    """Set a new SRC ratio.

    This allows step responses in the conversion ratio.
    Returns non zero on error.
    """
    return _lib.src_set_ratio(state, new_ratio) if state else None


def src_is_valid_ratio(ratio):
    """Return `True` if ratio is a valid conversion ratio, `False` otherwise.
    """
    return bool(_lib.src_is_valid_ratio(ratio))


@ffi.callback('src_callback_t')
def _src_input_callback(cb_data, data):
    """Internal callback function to be used with the callback API.

    Pulls the Python callback function from the handle contained in `cb_data`
    and calls it to fetch frames. Frames are converted to the format required by
    the API (float, interleaved channels). A reference to these data is kept
    internally.

    Returns
    -------
    frames : int
        The number of frames supplied.
    """
    cb_data = ffi.from_handle(cb_data)
    ret = cb_data['callback']()
    if ret is None:
        cb_data['last_input'] = None
        return 0  # No frames supplied
    input_data = _np.require(ret, requirements='C', dtype=_np.float32)
    input_frames, channels = _check_data(input_data)

    # Check whether the correct number of channels is supplied by user.
    if cb_data['channels'] != channels:
        raise ValueError('Invalid number of channels in callback.')

    # Store a reference of the input data to ensure it is still alive when
    # accessed by libsamplerate.
    cb_data['last_input'] = input_data

    data[0] = ffi.cast('float*', ffi.from_buffer(input_data))
    return input_frames


def src_callback_new(callback, converter_type, channels):
    """Initialisation for the callback based API.

    Parameters
    ----------
    callback : function
        Called whenever new frames are to be read. Must return a NumPy array
        of shape (num_frames, channels).
    converter_type : int
        Converter to be used.
    channels : int
        Number of channels.

    Returns
    -------
    state
        An anonymous pointer to the internal state of the converter.
    handle
        A CFFI handle to the callback data.
    error : int
        Error code.

    """
    cb_data = {'callback': callback, 'channels': channels}
    handle = ffi.new_handle(cb_data)
    error = ffi.new('int*')
    state = _lib.src_callback_new(_src_input_callback, converter_type,
                                  channels, error, handle)
    if state == ffi.NULL:
        return None, handle, error[0]
    return state, handle, error[0]


def src_callback_read(state, ratio, frames, data):
    """Read up to `frames` worth of data using the callback API.

    Returns
    -------
    frames : int
        Number of frames read or -1 on error.
    """
    data_ptr = ffi.cast('float*f', ffi.from_buffer(data))
    return _lib.src_callback_read(state, ratio, frames, data_ptr)


__libsamplerate_version__ = src_get_version()
if __libsamplerate_version__.startswith(lib_basename):
    __libsamplerate_version__ = __libsamplerate_version__[len(
        lib_basename) + 1:__libsamplerate_version__.find(' ')]
