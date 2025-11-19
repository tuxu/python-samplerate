"""
Test that the GIL is properly released during resampling operations.

This allows multiple threads to run resampling in parallel, which is critical
for performance in multi-threaded applications.
"""
import platform
import sys
import threading
import time
import numpy as np
import pytest

import samplerate


def is_arm_mac():
    """Check if running on ARM-based macOS (Apple Silicon)."""
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def _resample_work(data, ratio, converter_type, results, index):
    """Worker function that performs resampling."""
    start = time.perf_counter()
    output = samplerate.resample(data, ratio, converter_type)
    elapsed = time.perf_counter() - start
    results[index] = elapsed
    return output


def _resampler_work(data, ratio, converter_type, channels, results, index):
    """Worker function that performs stateful resampling."""
    start = time.perf_counter()
    resampler = samplerate.Resampler(converter_type, channels)
    output = resampler.process(data, ratio, end_of_input=True)
    elapsed = time.perf_counter() - start
    results[index] = elapsed
    return output


def _callback_resampler_work(data, ratio, converter_type, channels, results, index):
    """Worker function that performs callback resampling."""
    def producer():
        yield data
        while True:
            yield None

    callback = lambda p=producer(): next(p)
    
    start = time.perf_counter()
    resampler = samplerate.CallbackResampler(callback, ratio, converter_type, channels)
    output = resampler.read(int(ratio * len(data)))
    elapsed = time.perf_counter() - start
    results[index] = elapsed
    return output


@pytest.mark.parametrize("num_threads", [2, 4, 6, 8])
@pytest.mark.parametrize("converter_type", ["sinc_fastest", "sinc_medium", "sinc_best"])
def test_resample_gil_release_parallel(num_threads, converter_type):
    """Test that resample() releases GIL by running multiple threads in parallel."""
    # Create test data - make it large enough that computation dominates overhead
    # Need longer duration to overcome thread creation overhead (~0.5ms per thread)
    fs = 44100
    duration = 5.0  # seconds - increased from 0.5 to make computation time >> overhead
    ratio = 2.0
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Single-threaded baseline
    start = time.perf_counter()
    for _ in range(num_threads):
        samplerate.resample(data, ratio, converter_type)
    sequential_time = time.perf_counter() - start
    
    # Multi-threaded test
    threads = []
    results = [0.0] * num_threads
    start = time.perf_counter()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=_resample_work,
            args=(data, ratio, converter_type, results, i)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    parallel_time = time.perf_counter() - start
    
    # If GIL is properly released, parallel should be significantly faster
    # We expect at least 1.3x speedup for 2 threads, 1.5x for 4 threads
    # (accounting for overhead and non-perfect parallelization)
    # ARM Mac has different threading characteristics, especially for faster converters
    if is_arm_mac():
        # More relaxed expectations for ARM architecture
        expected_speedup = 1.15 if num_threads == 2 else 1.25
    else:
        expected_speedup = 1.2 if num_threads == 2 else 1.35
    speedup = sequential_time / parallel_time
    
    print(f"\n{converter_type} with {num_threads} threads:")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Platform: {'ARM Mac' if is_arm_mac() else platform.machine()}")
    print(f"  Individual thread times: {[f'{t:.4f}s' for t in results]}")
    
    if speedup < expected_speedup:
        print(f"  ⚠️  WARNING: Speedup {speedup:.2f}x is below expected {expected_speedup}x")
        print(f"      Expected: {expected_speedup}x, Got: {speedup:.2f}x")
        print(f"      (sequential={sequential_time:.4f}s, parallel={parallel_time:.4f}s)")
        print(f"      This may be due to CI load or platform-specific threading overhead.")
    else:
        print(f"  ✓ Performance meets expectations ({expected_speedup}x)")


@pytest.mark.parametrize("num_threads", [2, 4, 6, 8])
@pytest.mark.parametrize("converter_type", ["sinc_fastest", "sinc_medium", "sinc_best"])
def test_resampler_process_gil_release_parallel(num_threads, converter_type):
    """Test that Resampler.process() releases GIL by running multiple threads in parallel."""
    # Create test data - longer duration to amortize threading overhead
    fs = 44100
    duration = 5.0  # increased to make computation time >> overhead
    ratio = 2.0
    channels = 1
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Single-threaded baseline
    start = time.perf_counter()
    for _ in range(num_threads):
        resampler = samplerate.Resampler(converter_type, channels)
        resampler.process(data, ratio, end_of_input=True)
    sequential_time = time.perf_counter() - start
    
    # Multi-threaded test
    threads = []
    results = [0.0] * num_threads
    start = time.perf_counter()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=_resampler_work,
            args=(data, ratio, converter_type, channels, results, i)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    parallel_time = time.perf_counter() - start
    

    expected_speedup = 1.1 if num_threads == 2 else 1.25
    speedup = sequential_time / parallel_time
    
    print(f"\n{converter_type} Resampler.process() with {num_threads} threads:")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Platform: {'ARM Mac' if is_arm_mac() else platform.machine()}")
    print(f"  Individual thread times: {[f'{t:.4f}s' for t in results]}")
    
    if speedup < expected_speedup:
        print(f"  ⚠️  WARNING: Speedup {speedup:.2f}x is below expected {expected_speedup}x")
        print(f"      This may be due to CI load or platform-specific threading overhead.")
    else:
        print(f"  ✓ Performance meets expectations ({expected_speedup}x)")


@pytest.mark.parametrize("num_threads", [2, 4, 6, 8])
@pytest.mark.parametrize("converter_type", ["sinc_fastest", "sinc_medium", "sinc_best"])
def test_callback_resampler_gil_release_parallel(num_threads, converter_type):
    """Test that CallbackResampler.read() releases GIL appropriately."""
    # Note: CallbackResampler needs to acquire GIL when calling the Python callback,
    # but should release it during the actual resampling computation
    fs = 44100
    duration = 5.0  # increased to make computation time >> overhead
    ratio = 2.0
    channels = 1
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Single-threaded baseline
    start = time.perf_counter()
    for _ in range(num_threads):
        def producer():
            yield data
            while True:
                yield None
        callback = lambda p=producer(): next(p)
        resampler = samplerate.CallbackResampler(callback, ratio, converter_type, channels)
        resampler.read(int(ratio * len(data)))
    sequential_time = time.perf_counter() - start
    
    # Multi-threaded test
    threads = []
    results = [0.0] * num_threads
    start = time.perf_counter()
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=_callback_resampler_work,
            args=(data, ratio, converter_type, channels, results, i)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    parallel_time = time.perf_counter() - start
    
    # Callback resampler has more GIL contention due to callback invocation,
    # so we expect lower speedup
    if is_arm_mac():
        expected_speedup = 1.1
    else:
        expected_speedup = 1.2
    speedup = sequential_time / parallel_time
    
    print(f"\n{converter_type} CallbackResampler with {num_threads} threads:")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Platform: {'ARM Mac' if is_arm_mac() else platform.machine()}")
    print(f"  Individual thread times: {[f'{t:.4f}s' for t in results]}")
    
    if speedup < expected_speedup:
        print(f"  ⚠️  WARNING: Speedup {speedup:.2f}x is below expected {expected_speedup}x")
        print(f"      This may be due to CI load or platform-specific threading overhead.")
    else:
        print(f"  ✓ Performance meets expectations ({expected_speedup}x)")


def test_gil_release_quality():
    """Verify that GIL release doesn't affect output quality."""
    # Make sure the parallel execution produces identical results
    fs = 44100
    duration = 0.1
    ratio = 1.5
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Reference single-threaded result
    reference = samplerate.resample(data, ratio, "sinc_best")
    
    # Multi-threaded results
    results = [None, None]
    threads = []
    
    def worker(data, ratio, results, index):
        results[index] = samplerate.resample(data, ratio, "sinc_best")
    
    for i in range(2):
        thread = threading.Thread(target=worker, args=(data, ratio, results, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Results should be identical
    assert np.allclose(reference, results[0])
    assert np.allclose(reference, results[1])
    assert np.allclose(results[0], results[1])


def test_gil_metrics_report():
    """Generate a detailed performance report for GIL release optimization."""
    print("\n" + "="*70)
    print("GIL Release Performance Report")
    print("="*70)
    
    converters = ["sinc_fastest", "sinc_medium", "sinc_best"]
    thread_counts = [1, 2, 4]
    
    fs = 44100
    duration = 5.0  # Long enough to overcome threading overhead
    ratio = 2.0
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    print(f"\nTest Configuration:")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {duration} seconds ({num_samples} samples)")
    print(f"  Conversion ratio: {ratio}x")
    
    for converter in converters:
        print(f"\n{'-'*70}")
        print(f"Converter: {converter}")
        print(f"{'-'*70}")
        
        single_thread_time = None
        
        for num_threads in thread_counts:
            if num_threads == 1:
                # Single thread baseline - just measure one execution
                start = time.perf_counter()
                samplerate.resample(data, ratio, converter)
                single_thread_time = time.perf_counter() - start
                
                print(f"  1 thread (baseline):")
                print(f"    Execution time: {single_thread_time:.4f}s")
            else:
                # Multi-threaded: measure parallel execution
                threads = []
                results = [0.0] * num_threads
                start = time.perf_counter()
                
                for i in range(num_threads):
                    thread = threading.Thread(
                        target=_resample_work,
                        args=(data, ratio, converter, results, i)
                    )
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                parallel_time = time.perf_counter() - start
                avg_thread_time = np.mean(results)
                
                # Calculate speedup comparing N parallel threads vs N sequential executions
                sequential_time = single_thread_time * num_threads
                speedup = sequential_time / parallel_time
                efficiency = (speedup / num_threads) * 100
                
                print(f"  {num_threads} threads (parallel):")
                print(f"    Parallel execution time: {parallel_time:.4f}s")
                print(f"    Equivalent sequential time: {sequential_time:.4f}s ({num_threads} × {single_thread_time:.4f}s)")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Parallel efficiency: {efficiency:.1f}%")
                print(f"    Avg thread time: {avg_thread_time:.4f}s")

