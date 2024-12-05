import torch
import time
import numpy as np
import atexit

# Timer storage
cuda_timers = {}
timers = {}

class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        elapsed_time = self.start.elapsed_time(self.end)
        cuda_timers[self.timer_name].append(elapsed_time)

class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        elapsed_time = (self.end - self.start) * 1000.0  # Convert to milliseconds
        timers[self.timer_name].append(elapsed_time)

def print_timing_info():
    print('== Timing Statistics ==')
    for timer_name, timing_values in {**cuda_timers, **timers}.items():
        mean_time = np.mean(timing_values)
        if mean_time < 1000.0:
            print(f"{timer_name}: {mean_time:.2f} ms")
        else:
            print(f"{timer_name}: {mean_time / 1000:.2f} s")

# Register to print timings at program exit
atexit.register(print_timing_info)
