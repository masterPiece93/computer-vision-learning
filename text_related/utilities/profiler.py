import os
import time
import functools
import psutil
from utilities.colored import cprint, Styleit, Palette

Palette.register('NEON_ORANGE', '\033[38;2;255;95;31m')  # Register a custom color for orange

style_bold_neon = Styleit(Palette.NEON_ORANGE, B=True)  # Bold neon orange style
style_neon = Styleit(Palette.NEON_ORANGE)  # Regular neon orange style

def easy_profiler(num_runs=1):
    """
    Decorator to benchmark OCR execution time and memory footprint.
    
    Args:
        num_runs (int): Number of times to run the function for averaging results.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            
            # Record initial state
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            start_time = time.perf_counter()

            # Execute function (averaged over num_runs if requested)
            result = None
            for _ in range(num_runs):
                result = func(*args, **kwargs)

            end_time = time.perf_counter()
            mem_after = process.memory_info().rss / (1024 * 1024)   # MB

            # Metrics
            total_time = (end_time - start_time) / num_runs
            mem_used = mem_after - mem_before

            print(style_bold_neon >> "\n" + "=" * 45)
            print(style_neon >> f" ⏱️  BENCHMARK: {func.__name__}")
            print(style_neon >> "=" * 45)
            print(style_neon >> f" Avg Exec Time: {total_time * 1000:.2f} ms ({total_time:.4f} s)")
            print(style_neon >> f" RAM Delta:     {mem_used:+.2f} MB")
            print(style_neon >> f" Extracted Text: '{result}'")
            print(style_bold_neon >> "=" * 45 + "\n")
            print(style_neon >> f"\t - 📝  Function '{func.__name__}' executed {num_runs} time(s)")
            return result
        return wrapper
    return decorator
