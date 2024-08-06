import time


def high_precision_sleep(duration: float) -> None:
    """High precision sleep function.

    duration : float
        Duration to sleep in seconds.
    """
    if duration <= 0:
        return
    start_time = time.perf_counter()
    while True:
        elapsed_time = time.perf_counter() - start_time
        remaining_time = duration - elapsed_time
        if remaining_time <= 0:
            break
        if remaining_time >= 0.0002:
            time.sleep(remaining_time / 2)
