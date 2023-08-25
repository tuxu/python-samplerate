python-samplerate
=================

.. image:: https://img.shields.io/pypi/v/samplerate.svg
    :target: https://pypi.python.org/pypi/samplerate

.. image:: https://img.shields.io/pypi/l/samplerate.svg
    :target: https://pypi.python.org/pypi/samplerate

.. image:: https://img.shields.io/pypi/wheel/samplerate.svg
    :target: https://pypi.python.org/pypi/samplerate

.. image:: https://img.shields.io/pypi/pyversions/samplerate.svg
    :target: https://pypi.python.org/pypi/samplerate

.. image:: https://readthedocs.org/projects/python-samplerate/badge/?version=latest
   :target: http://python-samplerate.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


This is a wrapper around Erik de Castro Lopo's `libsamplerate`_ (aka Secret
Rabbit Code) for high-quality sample rate conversion.

It implements all three `APIs
<http://www.mega-nerd.com/libsamplerate/api.html>`_ available in
`libsamplerate`_:

* **Simple API**: for resampling a large chunk of data with a single library
  call
* **Full API**: for obtaining the resampled signal from successive chunks of
  data
* **Callback API**: like Full API, but input samples are provided by a callback
  function

The `libsamplerate`_ library is statically built together with the python bindings
using `pybind11 <https://github.com/pybind/pybind11/>`_.


Installation
------------

    $ pip install samplerate

Binary wheels of `libsamplerate`_ for macOS and Windows (64 bit) are available.
For other systems, a C++ 14 or above compiler is required to build the package.


Usage
-----

.. code-block:: python

   import numpy as np
   import samplerate

   # Synthesize data
   fs = 1000.
   t = np.arange(fs * 2) / fs
   input_data = np.sin(2 * np.pi * 5 * t)

   # Simple API
   ratio = 1.5
   converter = 'sinc_best'  # or 'sinc_fastest', ...
   output_data_simple = samplerate.resample(input_data, ratio, converter)

   # Full API
   resampler = samplerate.Resampler(converter, channels=1)
   output_data_full = resampler.process(input_data, ratio, end_of_input=True)

   # The result is the same for both APIs.
   assert np.allclose(output_data_simple, output_data_full)

   # See `samplerate.CallbackResampler` for the Callback API, or
   # `examples/play_modulation.py` for an example.

See ``samplerate.resample``, ``samplerate.Resampler``, and
``samplerate.CallbackResampler`` in the API documentation for details.


See also
--------

* `scikits.samplerate <https://pypi.python.org/pypi/scikits.samplerate>`_
  implements only the Simple API and uses `Cython <http://cython.org/>`_ for
  extern calls. The `resample` function of `scikits.samplerate` and this package
  share the same function signature for compatiblity.

* `resampy <https://github.com/bmcfee/resampy>`_: sample rate conversion in
  Python + Cython.


License
-------

This project is licensed under the `MIT license
<https://opensource.org/licenses/MIT>`_.

As of version 0.1.9, `libsamplerate`_ is licensed under the `2-clause BSD
license <https://opensource.org/licenses/BSD-2-Clause>`_.


.. _libsamplerate: http://www.mega-nerd.com/libsamplerate/
