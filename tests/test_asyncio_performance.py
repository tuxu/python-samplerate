"""
Test asyncio performance with resampling operations.

This demonstrates that CPU-bound resampling operations should use
executor-based async execution to avoid blocking the event loop,
and validates that GIL release allows true parallelism when using
ThreadPoolExecutor.

Event Loop Testing:
- Tests run with all available event loop implementations on the platform
- Windows: Tests with default asyncio and winloop (if installed)
- Unix/Linux/macOS: Tests with default asyncio and uvloop (if installed)
- Use the event_loop fixture to access the current loop type being tested
"""
import asyncio
import sys
import time
import numpy as np
import pytest

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import samplerate


def get_available_loop_types():
    """
    Get list of available event loop types.
    
    Returns:
        List of available loop types: always includes "default",
        plus "uvloop" (Unix only) and/or "winloop" (Windows only) if available.
    """
    available = ["default"]
    
    # uvloop only works on Unix-like systems
    if sys.platform != 'win32':
        try:
            import uvloop
            available.append("uvloop")
        except ImportError:
            pass
    
    # winloop only works on Windows
    if sys.platform == 'win32':
        try:
            import winloop
            available.append("winloop")
        except ImportError:
            pass
    
    return available


# Get available loop types for parameterization
AVAILABLE_LOOP_TYPES = get_available_loop_types()


@pytest.fixture(params=AVAILABLE_LOOP_TYPES)
def event_loop_policy(request):
    """
    Pytest fixture that provides different event loop policies.
    
    This allows pytest-asyncio to use uvloop, winloop, or default asyncio
    based on what's available on the platform.
    """
    loop_type = request.param
    
    if loop_type == "uvloop":
        import uvloop
        policy = uvloop.EventLoopPolicy()
    elif loop_type == "winloop":
        import winloop
        policy = winloop.EventLoopPolicy()
    else:
        policy = asyncio.DefaultEventLoopPolicy()
    
    # Store loop type for test output
    policy.loop_type_name = loop_type
    
    return policy


@pytest.fixture
def event_loop(event_loop_policy):
    """
    Override pytest-asyncio's event_loop fixture to use our custom policy.
    """
    asyncio.set_event_loop_policy(event_loop_policy)
    loop = event_loop_policy.new_event_loop()
    
    # Store loop type name on the loop for access in tests
    loop.loop_type_name = event_loop_policy.loop_type_name
    
    yield loop
    
    loop.close()
    asyncio.set_event_loop_policy(None)


async def resample_async(data, ratio, converter_type, executor=None):
    """Asynchronously resample data using an executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        samplerate.resample,
        data,
        ratio,
        converter_type
    )


async def resampler_process_async(data, ratio, converter_type, channels, executor=None):
    """Asynchronously resample using Resampler.process()."""
    def _process():
        resampler = samplerate.Resampler(converter_type, channels)
        return resampler.process(data, ratio, end_of_input=True)
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _process)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_concurrent", [2, 4, 8])
@pytest.mark.parametrize("converter_type", ["sinc_fastest", "sinc_medium", "sinc_best"])
async def test_asyncio_threadpool_parallel(event_loop, num_concurrent, converter_type):
    """Test async execution with ThreadPoolExecutor shows parallel speedup."""
    loop_type = event_loop.loop_type_name
    
    # Create test data
    fs = 44100
    duration = 5.0
    ratio = 2.0
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Sequential baseline - run tasks one at a time
    start = time.perf_counter()
    for _ in range(num_concurrent):
        samplerate.resample(data, ratio, converter_type)
    sequential_time = time.perf_counter() - start
    
    # Concurrent execution with ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=num_concurrent)
    try:
        start = time.perf_counter()
        tasks = [
            resample_async(data, ratio, converter_type, executor)
            for _ in range(num_concurrent)
        ]
        await asyncio.gather(*tasks)
        parallel_time = time.perf_counter() - start
    finally:
        executor.shutdown(wait=True)
    
    speedup = sequential_time / parallel_time
    expected_speedup = 1.3 if num_concurrent == 2 else 1.5
    
    print(f"\n{loop_type} loop - {converter_type} async with ThreadPoolExecutor ({num_concurrent} concurrent):")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    assert speedup >= expected_speedup, (
        f"Async with ThreadPoolExecutor should show speedup due to GIL release. "
        f"Expected {expected_speedup}x, got {speedup:.2f}x"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("converter_type", ["sinc_fastest"])
async def test_asyncio_no_executor_blocks(event_loop, converter_type):
    """Test that running CPU-bound work without executor blocks the event loop."""
    loop_type = event_loop.loop_type_name
    
    # This test demonstrates the WRONG way - blocking the event loop
    fs = 44100
    duration = 1.0
    ratio = 2.0
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # Run two tasks "concurrently" but without executor (blocks event loop)
    async def blocking_resample():
        # This blocks the event loop!
        return samplerate.resample(data, ratio, converter_type)
    
    start = time.perf_counter()
    task1 = asyncio.create_task(blocking_resample())
    task2 = asyncio.create_task(blocking_resample())
    await asyncio.gather(task1, task2)
    blocking_time = time.perf_counter() - start
    
    # Run with executor (proper async)
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        start = time.perf_counter()
        tasks = [
            resample_async(data, ratio, converter_type, executor)
            for _ in range(2)
        ]
        await asyncio.gather(*tasks)
        executor_time = time.perf_counter() - start
    finally:
        executor.shutdown(wait=True)
    
    print(f"\n{loop_type} loop - {converter_type} blocking vs executor:")
    print(f"  Without executor (blocks loop): {blocking_time:.4f}s")
    print(f"  With ThreadPoolExecutor: {executor_time:.4f}s")
    print(f"  Improvement: {blocking_time/executor_time:.2f}x")
    
    # Executor should be significantly faster (at least 1.3x due to parallelism)
    assert executor_time < blocking_time * 0.77, (
        "ThreadPoolExecutor should be faster than blocking the event loop"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("num_concurrent", [2, 4])
async def test_asyncio_processpool_comparison(event_loop, num_concurrent):
    """Compare ThreadPoolExecutor vs ProcessPoolExecutor for CPU-bound work."""
    loop_type = event_loop.loop_type_name
    
    # Note: ProcessPoolExecutor should be slower due to pickling overhead
    # for the large numpy arrays, even though it avoids GIL entirely
    
    fs = 44100
    duration = 2.0  # Shorter for process pool (slower due to overhead)
    ratio = 2.0
    converter_type = "sinc_fastest"
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    # ThreadPoolExecutor (benefits from GIL release)
    thread_executor = ThreadPoolExecutor(max_workers=num_concurrent)
    try:
        start = time.perf_counter()
        tasks = [
            resample_async(data, ratio, converter_type, thread_executor)
            for _ in range(num_concurrent)
        ]
        await asyncio.gather(*tasks)
        thread_time = time.perf_counter() - start
    finally:
        thread_executor.shutdown(wait=True)
    
    # ProcessPoolExecutor (no GIL but pickling overhead)
    process_executor = ProcessPoolExecutor(max_workers=num_concurrent)
    try:
        start = time.perf_counter()
        tasks = [
            resample_async(data, ratio, converter_type, process_executor)
            for _ in range(num_concurrent)
        ]
        await asyncio.gather(*tasks)
        process_time = time.perf_counter() - start
    finally:
        process_executor.shutdown(wait=True)
    
    print(f"\n{loop_type} loop - {num_concurrent} concurrent tasks - ThreadPool vs ProcessPool:")
    print(f"  ThreadPoolExecutor: {thread_time:.4f}s")
    print(f"  ProcessPoolExecutor: {process_time:.4f}s")
    print(f"  Ratio: {process_time/thread_time:.2f}x")
    
    # ThreadPool should be faster or comparable due to no pickling overhead
    # and GIL being properly released
    print(f"  → ThreadPool is {'faster' if thread_time < process_time else 'slower'}")
    print(f"    (GIL release makes ThreadPool competitive with ProcessPool)")


@pytest.mark.asyncio
async def test_asyncio_mixed_workload(event_loop):
    """Test mixing I/O and CPU-bound operations in async context."""
    loop_type = event_loop.loop_type_name
    
    fs = 44100
    duration = 1.0
    ratio = 2.0
    converter_type = "sinc_fastest"
    
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    async def io_task(delay):
        """Simulate I/O operation."""
        await asyncio.sleep(delay)
        return f"I/O completed after {delay}s"
    
    # Mix CPU-bound resampling with I/O tasks
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        start = time.perf_counter()
        results = await asyncio.gather(
            io_task(0.1),  # I/O task 1
            resample_async(data, ratio, converter_type, executor),  # CPU task 1
            io_task(0.2),  # I/O task 2
            resample_async(data, ratio, converter_type, executor),  # CPU task 2
            io_task(0.15),  # I/O task 3
        )
        total_time = time.perf_counter() - start
    finally:
        executor.shutdown(wait=True)
    
    print(f"\n{loop_type} loop - Mixed I/O and CPU workload:")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Tasks completed: {len(results)}")
    
    # Should complete faster than sequential execution
    # I/O: 0.1 + 0.2 + 0.15 = 0.45s
    # CPU: ~0.05s * 2 = ~0.1s
    # Sequential would be ~0.55s, parallel should be ~0.2-0.25s
    assert total_time < 0.35, (
        f"Mixed workload should complete faster than 0.35s, got {total_time:.4f}s"
    )


@pytest.mark.asyncio
async def test_asyncio_performance_report():
    """Generate comprehensive async performance report."""
    print("\n" + "="*70)
    print("Asyncio Performance Report")
    print("="*70)
    
    converters = ["sinc_fastest", "sinc_medium", "sinc_best"]
    concurrent_counts = [1, 2, 4]
    
    fs = 44100
    duration = 5.0
    ratio = 2.0
    num_samples = int(fs * duration)
    data = np.random.randn(num_samples).astype(np.float32)
    
    print(f"\nTest Configuration:")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {duration} seconds ({num_samples} samples)")
    print(f"  Conversion ratio: {ratio}x")
    print(f"  Executor: ThreadPoolExecutor")
    
    for converter in converters:
        print(f"\n{'-'*70}")
        print(f"Converter: {converter}")
        print(f"{'-'*70}")
        
        baseline_time = None
        
        for num_concurrent in concurrent_counts:
            if num_concurrent == 1:
                # Single task baseline
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    start = time.perf_counter()
                    await resample_async(data, ratio, converter, executor)
                    baseline_time = time.perf_counter() - start
                finally:
                    executor.shutdown(wait=True)
                
                print(f"  1 concurrent task (baseline):")
                print(f"    Execution time: {baseline_time:.4f}s")
            else:
                # Multiple concurrent tasks
                executor = ThreadPoolExecutor(max_workers=num_concurrent)
                try:
                    start = time.perf_counter()
                    tasks = [
                        resample_async(data, ratio, converter, executor)
                        for _ in range(num_concurrent)
                    ]
                    await asyncio.gather(*tasks)
                    parallel_time = time.perf_counter() - start
                finally:
                    executor.shutdown(wait=True)
                
                sequential_time = baseline_time * num_concurrent
                speedup = sequential_time / parallel_time
                efficiency = (speedup / num_concurrent) * 100
                
                print(f"  {num_concurrent} concurrent tasks:")
                print(f"    Parallel execution time: {parallel_time:.4f}s")
                print(f"    Equivalent sequential time: {sequential_time:.4f}s ({num_concurrent} × {baseline_time:.4f}s)")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Parallel efficiency: {efficiency:.1f}%")
