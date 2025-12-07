/*
 * Python bindings for libsamplerate
 * Copyright (C) 2023  Robin Scheibler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program.
 * If not, see <https://opensource.org/licenses/MIT>.
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samplerate.h>

#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#ifndef VERSION_INFO
#define VERSION_INFO "nightly"
#endif

// This value was empirically and somewhat arbitrarily chosen; increase it for further safety.
#define END_OF_INPUT_EXTRA_OUTPUT_FRAMES 10000

namespace py = pybind11;
using namespace pybind11::literals;

using callback_t =
    std::function<py::array_t<float, py::array::c_style | py::array::forcecast>(
        void)>;
using np_array_f32 =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace samplerate {

enum class ConverterType {
  sinc_best,
  sinc_medium,
  sinc_fastest,
  zero_order_hold,
  linear
};

class ResamplingException : public std::exception {
 public:
  explicit ResamplingException(int err_num) : message{src_strerror(err_num)} {}
  const char *what() const noexcept override { return message.c_str(); }

 private:
  std::string message = "";
};

int get_converter_type(const py::object &obj) {
  if (py::isinstance<py::str>(obj)) {
    py::str py_s = obj;
    std::string s = static_cast<std::string>(py_s);
    if (s.compare("sinc_best") == 0) {
      return 0;
    } else if (s.compare("sinc_medium") == 0) {
      return 1;
    } else if (s.compare("sinc_fastest") == 0) {
      return 2;
    } else if (s.compare("zero_order_hold") == 0) {
      return 3;
    } else if (s.compare("linear") == 0) {
      return 4;
    }
  } else if (py::isinstance<py::int_>(obj)) {
    py::int_ val = obj;
    return static_cast<int>(val);
  } else if (py::isinstance<ConverterType>(obj)) {
    py::int_ c = obj.attr("value");
    return static_cast<int>(c);
  }

  throw std::domain_error("Unsupported converter type");
  return -1;
}

void error_handler(int errnum) {
  if (errnum > 0 && errnum < 24) {
    throw ResamplingException(errnum);
  } else if (errnum != 0) {  // the zero case is excluded as it is not an error
    // this will throw a segmentation fault if we call src_strerror here
    // also, these should never happen
    throw std::runtime_error("libsamplerate raised an unknown error code");
  }
}

class Resampler {
 private:
  SRC_STATE *_state = nullptr;

 public:
  int _converter_type = 0;
  int _channels = 0;

 public:
  Resampler(const py::object &converter_type, int channels)
      : _converter_type(get_converter_type(converter_type)),
        _channels(channels) {
    int _err_num = 0;
    _state = src_new(_converter_type, _channels, &_err_num);
    error_handler(_err_num);
  }

  // copy constructor
  Resampler(const Resampler &r)
      : _converter_type(r._converter_type), _channels(r._channels) {
    int _err_num = 0;
    _state = src_clone(r._state, &_err_num);
    error_handler(_err_num);
  }

  // move constructor
  Resampler(Resampler &&r)
      : _state(r._state),
        _converter_type(r._converter_type),
        _channels(r._channels) {
    r._state = nullptr;
    r._converter_type = 0;
    r._channels = 0;
  }

  ~Resampler() { src_delete(_state); }  // src_delete handles nullptr case

  py::array_t<float, py::array::c_style> process(
      py::array_t<float, py::array::c_style | py::array::forcecast> input,
      double sr_ratio, bool end_of_input) {
    // accessors for the arrays
    py::buffer_info inbuf = input.request();

    // set the number of channels
    int channels = 1;
    if (inbuf.ndim == 2)
      channels = inbuf.shape[1];
    else if (inbuf.ndim > 2)
      throw std::domain_error("Input array should have at most 2 dimensions");

    if (channels != _channels || channels == 0)
      throw std::domain_error("Invalid number of channels in input data.");

    // Add a "fudge factor" to the size. This is because the actual number of
    // output samples generated on the last call when input is terminated can
    // be more than the expected number of output samples during mid-stream
    // steady-state processing. (Also, when the stream is started, the number
    // of output samples generated will generally be zero or otherwise less
    // than the number of samples in mid-stream processing.)
    const auto new_size =
        static_cast<size_t>(std::ceil(inbuf.shape[0] * sr_ratio))
        + END_OF_INPUT_EXTRA_OUTPUT_FRAMES;

    // allocate output array
    std::vector<size_t> out_shape{new_size};
    if (inbuf.ndim == 2) out_shape.push_back(static_cast<size_t>(channels));
    auto output = py::array_t<float, py::array::c_style>(out_shape);
    py::buffer_info outbuf = output.request();

    // libsamplerate struct
    SRC_DATA src_data = {
        static_cast<float *>(inbuf.ptr),   // data_in
        static_cast<float *>(outbuf.ptr),  // data_out
        inbuf.shape[0],                    // input_frames
        long(new_size),                    // output_frames
        0,             // input_frames_used, filled by libsamplerate
        0,             // output_frames_gen, filled by libsamplerate
        end_of_input,  // end_of_input, not used by src_simple ?
        sr_ratio       // src_ratio, sampling rate conversion ratio
    };

    // Release GIL for the entire resampling operation
    int err_code;
    long output_frames_gen;
    {
      py::gil_scoped_release release;
      err_code = src_process(_state, &src_data);
      output_frames_gen = src_data.output_frames_gen;
    }
    error_handler(err_code);

    // create a shorter view of the array
    if ((size_t)output_frames_gen < new_size) {
      out_shape[0] = output_frames_gen;
      output.resize(out_shape);
    } else if ((size_t)output_frames_gen >= new_size) {
      // This means our fudge factor is too small.
      throw std::runtime_error("Generated more output samples than expected!");
    }

    return output;
  }

  void set_ratio(double new_ratio) {
    error_handler(src_set_ratio(_state, new_ratio));
  }

  void reset() { error_handler(src_reset(_state)); }

  Resampler clone() const { return Resampler(*this); }
};

namespace {

long the_callback_func(void *cb_data, float **data);

}  // namespace

class CallbackResampler {
 private:
  SRC_STATE *_state = nullptr;
  callback_t _callback = nullptr;
  np_array_f32 _current_buffer;
  size_t _buffer_ndim = 0;
  std::string _callback_error_msg = "";

 public:
  double _ratio = 0.0;
  int _converter_type = 0;
  size_t _channels = 0;

 private:
  void _create() {
    int _err_num = 0;
    _state = src_callback_new(the_callback_func, _converter_type, (int)_channels,
                              &_err_num, static_cast<void *>(this));
    if (_state == nullptr) error_handler(_err_num);
  }

  void _destroy() {
    if (_state != nullptr) {
      src_delete(_state);
      _state = nullptr;
    }
  }

 public:
  CallbackResampler(const callback_t &callback_func, double ratio,
                    const py::object &converter_type, size_t channels)
      : _callback(callback_func),
        _ratio(ratio),
        _converter_type(get_converter_type(converter_type)),
        _channels(channels) {
    _create();
  }

  // copy constructor
  CallbackResampler(const CallbackResampler &r)
      : _callback(r._callback),
        _ratio(r._ratio),
        _converter_type(r._converter_type),
        _channels(r._channels) {
    int _err_num = 0;
    _state = src_clone(r._state, &_err_num);
    if (_state == nullptr) error_handler(_err_num);
  }

  // move constructor
  CallbackResampler(CallbackResampler &&r)
      : _state(r._state),
        _callback(r._callback),
        _current_buffer(std::move(r._current_buffer)),
        _buffer_ndim(r._buffer_ndim),
        _callback_error_msg(std::move(r._callback_error_msg)),
        _ratio(r._ratio),
        _converter_type(r._converter_type),
        _channels(r._channels) {
    r._state = nullptr;
    r._callback = nullptr;
    r._buffer_ndim = 0;
    r._ratio = 0.0;
    r._converter_type = 0;
    r._channels = 0;
  }

  ~CallbackResampler() { _destroy(); }

  void set_buffer(const np_array_f32 &new_buf) { _current_buffer = new_buf; }
  size_t get_channels() { return _channels; }
  void set_callback_error(const std::string &error_msg) {
    _callback_error_msg = error_msg;
  }
  std::string get_callback_error() const { return _callback_error_msg; }
  void clear_callback_error() { _callback_error_msg = ""; }

  np_array_f32 callback(void) {
    auto input = _callback();

    auto inbuf = input.request();
    if (_buffer_ndim == 0) _buffer_ndim = inbuf.ndim;

    _current_buffer = input;
    return input;
  }

  py::array_t<float, py::array::c_style> read(size_t frames) {
    // allocate output array
    std::vector<size_t> out_shape{frames, _channels};
    auto output = py::array_t<float, py::array::c_style>(out_shape);
    py::buffer_info outbuf = output.request();

    if (_state == nullptr) _create();

    // clear any previous callback error
    clear_callback_error();

    // read from the callback - note: GIL is managed by the_callback_func
    // which acquires it only when calling the Python callback
    size_t output_frames_gen = 0;
    int err_code = 0;
    {
      py::gil_scoped_release release;
      output_frames_gen = src_callback_read(_state, _ratio, (long)frames,
                                            static_cast<float *>(outbuf.ptr));
      // Get error code while GIL is released
      if (output_frames_gen == 0) {
        err_code = src_error(_state);
      }
    }

    // check if callback had an error
    std::string callback_error = get_callback_error();
    if (!callback_error.empty()) {
      throw std::domain_error(callback_error);
    }

    // check error status
    if (output_frames_gen == 0) {
      error_handler(err_code);
    }

    // if there is only one channel and the input array had only on dimension
    // we also output a 1D array
    if (_channels == 1 && _buffer_ndim == 1) {
      out_shape.pop_back();
      output = py::array_t<float, py::array::c_style>(
          out_shape, static_cast<float *>(outbuf.ptr));
    }

    // create a shorter view of the array
    if (output_frames_gen < frames) {
      out_shape[0] = output_frames_gen;
      output.resize(out_shape);
    }

    return output;
  }

  void set_starting_ratio(double new_ratio) {
    error_handler(src_set_ratio(_state, new_ratio));
    _ratio = new_ratio;
  }

  void reset() { error_handler(src_reset(_state)); }

  CallbackResampler clone() const { return CallbackResampler(*this); }
  CallbackResampler &__enter__() { return *this; }
  void __exit__(const py::object &/*exc_type*/, const py::object &/*exc*/,
                const py::object &/*exc_tb*/) {
    _destroy();
  }
};

namespace {

long the_callback_func(void *cb_data, float **data) {
  CallbackResampler *cb = static_cast<CallbackResampler *>(cb_data);
  int cb_channels = cb->get_channels();

  py::buffer_info inbuf;
  {
    py::gil_scoped_acquire acquire;

    // get the data as a numpy array
    auto input = cb->callback();
    inbuf = input.request();
  }

  // end of stream is signaled by a None, which is cast to a ndarray with ndim
  // == 0
  if (inbuf.ndim == 0) return 0;

  // set the number of channels
  int channels = 1;
  if (inbuf.ndim == 2)
    channels = inbuf.shape[1];
  else if (inbuf.ndim > 2) {
    // Cannot throw exception in C callback - store error and return 0
    cb->set_callback_error("Input array should have at most 2 dimensions");
    return 0;
  }

  if (channels != cb_channels || channels == 0) {
    // Cannot throw exception in C callback - store error and return 0
    cb->set_callback_error("Invalid number of channels in input data.");
    return 0;
  }

  *data = static_cast<float *>(inbuf.ptr);

  return (long)inbuf.shape[0];
}

}  // namespace

py::array_t<float, py::array::c_style> resample(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &input,
    double sr_ratio, const py::object &converter_type, bool verbose) {
  // input array has shape (n_samples, n_channels)
  int converter_type_int = get_converter_type(converter_type);

  // accessors for the arrays
  py::buffer_info inbuf = input.request();

  // set the number of channels
  int channels = 1;
  if (inbuf.ndim == 2)
    channels = inbuf.shape[1];
  else if (inbuf.ndim > 2)
    throw std::domain_error("Input array should have at most 2 dimensions");

  if (channels == 0)
    throw std::domain_error("Invalid number of channels (0) in input data.");

  // Add buffer space to match Resampler.process() behavior with end_of_input=True
  // src_simple internally behaves like end_of_input=True, so it may generate
  // extra samples from buffer flushing, especially for certain converters
  const auto new_size =
      static_cast<size_t>(std::ceil(inbuf.shape[0] * sr_ratio))
      + END_OF_INPUT_EXTRA_OUTPUT_FRAMES;

  // allocate output array
  std::vector<size_t> out_shape{new_size};
  if (inbuf.ndim == 2) out_shape.push_back(static_cast<size_t>(channels));
  auto output = py::array_t<float, py::array::c_style>(out_shape);
  py::buffer_info outbuf = output.request();

  // libsamplerate struct
  SRC_DATA src_data = {
      static_cast<float *>(inbuf.ptr),   // data_in
      static_cast<float *>(outbuf.ptr),  // data_out
      inbuf.shape[0],                    // input_frames
      long(new_size),                    // output_frames
      0,        // input_frames_used, filled by libsamplerate
      0,        // output_frames_gen, filled by libsamplerate
      0,        // end_of_input, not used by src_simple ?
      sr_ratio  // src_ratio, sampling rate conversion ratio
  };

  // Release GIL for the entire resampling operation
  int err_code;
  long output_frames_gen;
  long input_frames_used;
  {
    py::gil_scoped_release release;
    err_code = src_simple(&src_data, converter_type_int, channels);
    output_frames_gen = src_data.output_frames_gen;
    input_frames_used = src_data.input_frames_used;
  }
  error_handler(err_code);

  // create a shorter view of the array
  if ((size_t)output_frames_gen < new_size) {
    out_shape[0] = output_frames_gen;
    output.resize(out_shape);
  } else if ((size_t)output_frames_gen >= new_size) {
    // This means our fudge factor is too small.
    throw std::runtime_error("Generated more output samples than expected!");
  }

  if (verbose) {
    py::print("samplerate info:");
    py::print(input_frames_used, " input frames used");
    py::print(output_frames_gen, " output frames generated");
  }

  return output;
}

}  // namespace samplerate

namespace sr = samplerate;

PYBIND11_MODULE(samplerate, m) {
  m.doc() =
      "A simple python wrapper library around libsamplerate";  // optional
                                                               // module
                                                               // docstring
  m.attr("__version__") = VERSION_INFO;
  m.attr("__libsamplerate_version__") = LIBSAMPLERATE_VERSION;

  auto m_exceptions = m.def_submodule(
      "exceptions", "Sub-module containing sampling exceptions");
  auto m_converters = m.def_submodule(
      "converters", "Sub-module containing the samplerate converters");
  auto m_internals = m.def_submodule("_internals", "Internal helper functions");

  // give access to this function for testing
  m_internals.def(
      "get_converter_type", &sr::get_converter_type,
      "Convert python object to integer of converter tpe or raise an error "
      "if illegal");

  m_internals.def(
      "error_handler", &sr::error_handler,
      "A function to translate libsamplerate error codes into exceptions");

  py::register_exception<sr::ResamplingException>(
      m_exceptions, "ResamplingError", PyExc_RuntimeError);

  m_converters.def("resample", &sr::resample, R"mydelimiter(
    Resample the signal in `input_data` at once.

    Parameters
    ----------
    input_data : ndarray
        Input data.
        Input data with one or more channels is represented as a 2D array of shape
        (`num_frames`, `num_channels`).
        A single channel can be provided as a 1D array of `num_frames` length.
        For use with `libsamplerate`, `input_data`
        is converted to 32-bit float and C (row-major) memory order.
    ratio : float
        Conversion ratio = output sample rate / input sample rate.
    converter_type : ConverterType, str, or int
        Sample rate converter (default: `sinc_best`).
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
  )mydelimiter",
                   "input"_a, "ratio"_a, "converter_type"_a = "sinc_best",
                   "verbose"_a = false);

  py::class_<sr::Resampler>(m_converters, "Resampler", R"mydelimiter(
    Resampler.

    Parameters
    ----------
    converter_type : ConverterType, str, or int
        Sample rate converter (default: `sinc_best`).
    num_channels : int
        Number of channels.
  )mydelimiter")
      .def(py::init<const py::object &, int>(),
           "converter_type"_a = "sinc_best", "channels"_a = 1)
      .def(py::init<sr::Resampler>())
      .def("process", &sr::Resampler::process, R"mydelimiter(
        Resample the signal in `input_data`.

        Parameters
        ----------
        input_data : ndarray
            Input data.
            Input data with one or more channels is represented as a 2D array of shape
            (`num_frames`, `num_channels`).
            A single channel can be provided as a 1D array of `num_frames` length.
            For use with `libsamplerate`, `input_data` is converted to 32-bit float and
            C (row-major) memory order.
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
      )mydelimiter",
           "input"_a, "ratio"_a, "end_of_input"_a = false)
      .def("reset", &sr::Resampler::reset, "Reset internal state.")
      .def("set_ratio", &sr::Resampler::set_ratio,
           "Set a new conversion ratio immediately.")
      .def("clone", &sr::Resampler::clone,
           "Creates a copy of the resampler object with the same internal "
           "state.")
      .def_readonly("converter_type", &sr::Resampler::_converter_type,
                    "Converter type.")
      .def_readonly("channels", &sr::Resampler::_channels,
                    "Number of channels.");

  py::class_<sr::CallbackResampler>(m_converters, "CallbackResampler",
                                    R"mydelimiter(
    CallbackResampler.

    Parameters
    ----------
    callback : function
        Function that returns new frames on each call, or `None` otherwise.
        Input data with one or more channels is represented as a 2D array of shape
        (`num_frames`, `num_channels`).
        A single channel can be provided as a 1D array of `num_frames` length.
        For use with `libsamplerate`, `input_data` is converted to 32-bit float and
        C (row-major) memory order.
    ratio : float
        Conversion ratio = output sample rate / input sample rate.
    converter_type : ConverterType, str, or int
        Sample rate converter.
    channels : int
        Number of channels.
    )mydelimiter")
      .def(py::init<const callback_t &, double, const py::object &, int>(),
           "callback"_a, "ratio"_a, "converter_type"_a = "sinc_best",
           "channels"_a = 1)
      .def(py::init<sr::CallbackResampler>())
      .def("read", &sr::CallbackResampler::read, R"mydelimiter(
            Read a number of frames from the resampler.

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
           )mydelimiter",
           "num_frames"_a)
      .def("reset", &sr::CallbackResampler::reset, "Reset state.")
      .def("set_starting_ratio", &sr::CallbackResampler::set_starting_ratio,
           "Set the starting conversion ratio for the next `read` call.")
      .def("clone", &sr::CallbackResampler::clone,
           "Create a copy of the resampler object.")
      .def("__enter__", &sr::CallbackResampler::__enter__,
           py::return_value_policy::reference_internal)
      .def("__exit__", &sr::CallbackResampler::__exit__)
      .def_readwrite(
          "ratio", &sr::CallbackResampler::_ratio,
          "Conversion ratio = output sample rate / input sample rate.")
      .def_readonly("converter_type", &sr::CallbackResampler::_converter_type,
                    "Converter type.")
      .def_readonly("channels", &sr::CallbackResampler::_channels,
                    "Number of channels.");

  py::enum_<sr::ConverterType>(m_converters, "ConverterType", R"mydelimiter(
      Enum of samplerate converter types.

      Pass any of the members, or their string or value representation, as
      ``converter_type`` in the resamplers.
    )mydelimiter")
      .value("sinc_best", sr::ConverterType::sinc_best)
      .value("sinc_medium", sr::ConverterType::sinc_medium)
      .value("sinc_fastest", sr::ConverterType::sinc_fastest)
      .value("zero_order_hold", sr::ConverterType::zero_order_hold)
      .value("linear", sr::ConverterType::linear)
      .export_values();

  // Convenience imports
  m.attr("ResamplingError") = m_exceptions.attr("ResamplingError");
  m.attr("resample") = m_converters.attr("resample");
  m.attr("CallbackResampler") = m_converters.attr("CallbackResampler");
  m.attr("Resampler") = m_converters.attr("Resampler");
  m.attr("ConverterType") = m_converters.attr("ConverterType");
}
