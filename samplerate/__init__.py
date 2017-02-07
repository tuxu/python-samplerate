"""Python bindings for libsamplerate based on CFFI and NumPy.

"""
# Versioning
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .lowlevel import __libsamplerate_version__

# Convenience imports
from .exceptions import SampleRateError
from .converters import resample
from .converters import SampleRateConverter
from .converters import resampling_callback

(SRC_SINC_BEST_QUALITY, SRC_SINC_MEDIUM_QUALITY,
 SRC_SINC_FASTEST, SRC_ZERO_ORDER_HOLD,
 SRC_LINEAR) = range(5)
