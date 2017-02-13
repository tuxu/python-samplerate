#!/usr/bin/env python
"""Demonstrate realtime audio resampling and playback using the callback API.

A carrier frequency is modulated by a sine wave, and the resulting signal is
played back on the default sound output. During playback, the modulation signal
is generated at source samplerate, then resampled to target samplerate, and
mixed onto the carrier.
"""
from __future__ import print_function, division
import numpy as np
import sounddevice as sd
import samplerate as sr

source_samplerate = 3600
target_samplerate = 44100
converter_type = 'sinc_fastest'

params = {
    'mod_amplitude': 1,  # Modulation amplitude (Hz)
    'mod_frequency': 1,  # Modulation frequency (Hz)
    'fm_gain': 20,  # FM gain (Hz/Hz)
    'output_volume': 0.1,  # Output volume
    'carrier_frequency': 500,  # Carrier frequency (Hz)
}


def get_input_callback(samplerate, params, num_samples=256):
    """Return a function that produces samples of a sine.

    Parameters
    ----------
    samplerate : float
        The sample rate.
    params : dict
        Parameters for FM generation.
    num_samples : int, optional
        Number of samples to be generated on each call.
    """
    amplitude = params['mod_amplitude']
    frequency = params['mod_frequency']

    def producer():
        """Generate samples.

        Yields
        ------
        samples : ndarray
            A number of samples (`num_samples`) of the sine.
        """
        start_time = 0
        while True:
            time = start_time + np.arange(num_samples) / samplerate
            start_time += num_samples / samplerate
            output = amplitude * np.cos(2 * np.pi * frequency * time)
            yield output

    return lambda p=producer(): next(p)


def get_playback_callback(resampler, samplerate, params):
    """Return a sound playback callback.

    Parameters
    ----------
    resampler
        The resampler from which samples are read.
    samplerate : float
        The sample rate.
    params : dict
        Parameters for FM generation.
    """

    def callback(outdata, frames, time, _):
        """Playback callback.

        Read samples from the resampler and modulate them onto a carrier
        frequency.
        """
        last_fmphase = getattr(callback, 'last_fmphase', 0)
        df = params['fm_gain'] * resampler.read(frames)
        df = np.pad(df, (0, frames - len(df)), mode='constant')
        t = time.outputBufferDacTime + np.arange(frames) / samplerate
        phase = 2 * np.pi * params['carrier_frequency'] * t
        fmphase = last_fmphase + 2 * np.pi * np.cumsum(df) / samplerate
        outdata[:, 0] = params['output_volume'] * np.cos(phase + fmphase)
        callback.last_fmphase = fmphase[-1]

    return callback


def main(source_samplerate, target_samplerate, params, converter_type):
    """Setup the resampling and audio output callbacks and start playback."""
    from time import sleep

    ratio = target_samplerate / source_samplerate

    with sr.CallbackResampler(get_input_callback(source_samplerate, params),
                              ratio, converter_type) as resampler, \
            sd.OutputStream(channels=1, samplerate=target_samplerate,
                            callback=get_playback_callback(
                                resampler, target_samplerate, params)):
        print("Playing back...  Ctrl+C to stop.")
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            print("Aborting.")


if __name__ == '__main__':
    main(
        source_samplerate=source_samplerate,
        target_samplerate=target_samplerate,
        params=params,
        converter_type=converter_type)
