"""Converters

"""
from __future__ import print_function, division
from enum import Enum
import numpy as np


class ConverterType(Enum):
    """Enum of samplerate converter types.

    Pass any of the members, or their string or value representation, as
    ``converter_type`` in the resamplers.
    """
    sinc_best = 0
    sinc_medium = 1
    sinc_fastest = 2
    zero_order_hold = 3
    linear = 4


def _get_converter_type(identifier):
    """Return the converter type for `identifier`."""
    if isinstance(identifier, str):
        return ConverterType[identifier]
    if isinstance(identifier, ConverterType):
        return identifier
    return ConverterType(identifier)


def resample(input_data, ratio, converter_type='sinc_best', verbose=False):
    """Resample the signal in `input_data` at once.

    Parameters
    ----------
    input_data : ndarray
        Input data. A single channel is provided as a 1D array of `num_frames` length.
        Input data with several channels is represented as a 2D array of shape
        (`num_frames`, `num_channels`). For use with `libsamplerate`, `input_data`
        is converted to 32-bit float and C (row-major) memory order.
    ratio : float
        Conversion ratio = output sample rate / input sample rate.
    converter_type : ConverterType, str, or int
        Sample rate converter.
    verbose : bool
        If `True`, print additional information about the conversion.

    Returns
    -------
    output_data : ndarray
        Resampled input data.

    Note
    ----
    If samples are to be processed in chunks, `Resampler` and
    `CallbackResampler` will provide better results and allow for variable
    conversion ratios.
    """
    from samplerate.lowlevel import src_simple
    from samplerate.exceptions import ResamplingError

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
    converter_type = _get_converter_type(converter_type)

    (error, input_frames_used, output_frames_gen) \
        = src_simple(input_data, output_data, ratio,
                     converter_type.value, channels)

    if error != 0:
        raise ResamplingError(error)

    if verbose:
        info = ('samplerate info:\n'
                '{} input frames used\n'
                '{} output frames generated\n'
                .format(input_frames_used, output_frames_gen))
        print(info)

    return (output_data[:output_frames_gen, :]
            if channels > 1 else output_data[:output_frames_gen])


class Resampler(object):
    """Resampler.

    Parameters
    ----------
    converter_type : ConverterType, str, or int
        Sample rate converter.
    num_channels : int
        Number of channels.
    """
    def __init__(self, converter_type='sinc_fastest', channels=1):
        from samplerate.lowlevel import ffi, src_new, src_delete
        from samplerate.exceptions import ResamplingError

        converter_type = _get_converter_type(converter_type)
        state, error = src_new(converter_type.value, channels)
        self._state = ffi.gc(state, src_delete)
        self._converter_type = converter_type
        self._channels = channels
        if error != 0:
            raise ResamplingError(error)

    @property
    def converter_type(self):
        """Converter type."""
        return self._converter_type

    @property
    def channels(self):
        """Number of channels."""
        return self._channels

    def reset(self):
        """Reset internal state."""
        from samplerate.lowlevel import src_reset
        return src_reset(self._state)

    def set_ratio(self, new_ratio):
        """Set a new conversion ratio immediately."""
        from samplerate.lowlevel import src_set_ratio
        return src_set_ratio(self._state, new_ratio)

    def process(self, input_data, ratio, end_of_input=False, verbose=False):
        """Resample the signal in `input_data`.

        Parameters
        ----------
        input_data : ndarray
            Input data. A single channel is provided as a 1D array of `num_frames` length.
            Input data with several channels is represented as a 2D array of shape
            (`num_frames`, `num_channels`). For use with `libsamplerate`, `input_data`
            is converted to 32-bit float and C (row-major) memory order.
        ratio : float
            Conversion ratio = output sample rate / input sample rate.
        end_of_input : int
            Set to `True` if no more data is available, or to `False` otherwise.
        verbose : bool
            If `True`, print additional information about the conversion.

        Returns
        -------
        output_data : ndarray
            Resampled input data.
        """
        from samplerate.lowlevel import src_process
        from samplerate.exceptions import ResamplingError

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

        (error, input_frames_used, output_frames_gen) = src_process(
            self._state, input_data, output_data, ratio, end_of_input)

        if error != 0:
            raise ResamplingError(error)

        if verbose:
            info = ('samplerate info:\n'
                    '{} input frames used\n'
                    '{} output frames generated\n'
                    .format(input_frames_used, output_frames_gen))
            print(info)

        return (output_data[:output_frames_gen, :]
                if channels > 1 else output_data[:output_frames_gen])


class CallbackResampler(object):
    """CallbackResampler.

    Parameters
    ----------
    callback : function
        Function that returns new frames on each call, or `None` otherwise.
        A single channel is provided as a 1D array of `num_frames` length.
        Input data with several channels is represented as a 2D array of shape
        (`num_frames`, `num_channels`). For use with `libsamplerate`, `input_data`
        is converted to 32-bit float and C (row-major) memory order.
    ratio : float
        Conversion ratio = output sample rate / input sample rate.
    converter_type : ConverterType, str, or int
        Sample rate converter.
    channels : int
        Number of channels.
    """
    def __init__(self, callback, ratio, converter_type='sinc_fastest',
                 channels=1):
        if channels < 1:
            raise ValueError('Invalid number of channels.')
        self._callback = callback
        self._ratio = ratio
        self._converter_type = _get_converter_type(converter_type)
        self._channels = channels
        self._state = None
        self._handle = None
        self._create()

    def _create(self):
        """Create new callback resampler."""
        from samplerate.lowlevel import ffi, src_callback_new, src_delete
        from samplerate.exceptions import ResamplingError

        state, handle, error = src_callback_new(
            self._callback, self._converter_type.value, self._channels)
        if error != 0:
            raise ResamplingError(error)
        self._state = ffi.gc(state, src_delete)
        self._handle = handle

    def _destroy(self):
        """Destroy resampler state."""
        if self._state:
            self._state = None
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._destroy()

    def set_starting_ratio(self, ratio):
        """ Set the starting conversion ratio for the next `read` call. """
        from samplerate.lowlevel import src_set_ratio
        if self._state is None:
            self._create()
        src_set_ratio(self._state, ratio)
        self.ratio = ratio

    def reset(self):
        """Reset state."""
        from samplerate.lowlevel import src_reset
        if self._state is None:
            self._create()
        src_reset(self._state)

    @property
    def ratio(self):
        """Conversion ratio = output sample rate / input sample rate."""
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self._ratio = ratio

    def read(self, num_frames):
        """Read a number of frames from the resampler.

        Parameters
        ----------
        num_frames : int
            Number of frames to read.

        Returns
        -------
        output_data : ndarray
            Resampled frames as a (`num_output_frames`, `num_channels`) or
            (`num_output_frames`,) array. Note that this may return fewer frames
            than requested, for example when no more input is available.
        """
        from samplerate.lowlevel import src_callback_read, src_error
        from samplerate.exceptions import ResamplingError

        if self._state is None:
            self._create()
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
                raise ResamplingError(error)

        return (output_data[:ret, :]
                if self._channels > 1 else output_data[:ret])
