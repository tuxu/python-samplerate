"""Python bindings for libsamplerate based on CFFI and NumPy.

"""
# Versioning
from samplerate._version import get_versions
__version__ = get_versions()['version']
del get_versions

from samplerate.lowlevel import __libsamplerate_version__

# Convenience imports
from samplerate.exceptions import ResamplingError
from samplerate.converters import resample
from samplerate.converters import Resampler
from samplerate.converters import CallbackResampler
