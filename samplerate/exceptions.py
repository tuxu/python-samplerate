"""Exceptions

"""


class ResamplingError(RuntimeError):
    """Runtime exception of libsamplerate"""

    def __init__(self, error):
        from samplerate.lowlevel import src_strerror
        message = 'libsamplerate error #{}: {}'.format(error,
                                                       src_strerror(error))
        super(ResamplingError, self).__init__(message)
        self.error = error
