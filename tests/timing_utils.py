import time
import asyncio
from contextlib import asynccontextmanager


class TimingStats:
    def __init__(self):
        self.measurements = {}

    def add_measurement(self, name: str, duration: float):
        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration)

    def report(self):
        print("\n=== Timing Report ===")
        for name, durations in self.measurements.items():
            avg = sum(durations) / len(durations)
            total = sum(durations)
            count = len(durations)
            print(
                f"{name:.<40} Count: {count:>3} | Total: {total:7.3f}s | Avg: {avg:7.3f}s"
            )
        print("=====================\n")


stats = TimingStats()


@asynccontextmanager
async def time_it(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        stats.add_measurement(name, duration)


def time_func(name: str):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                async with time_it(name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    stats.add_measurement(name, duration)

            return sync_wrapper

    return decorator
