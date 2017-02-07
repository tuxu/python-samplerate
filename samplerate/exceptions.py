"""Exceptions

"""

class SampleRateError(RuntimeError):
    """Runtime exception of libsamplerate"""

    def __init__(self, error):
        from .lowlevel import src_strerror
        message = 'libsamplerate error #{}: {}'.format(
            error, src_strerror(error)
        )
        super(SampleRateError, self).__init__(message)
        self.error = error
