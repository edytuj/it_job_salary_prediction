import statistics
import time

from config.settings import settings
from model.model_loader import get_model
from prediction.sample_data import get_dummy_input


def measure(func):
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000  # ms


def run_benchmark(n_runs=20, warmup_runs=5, model=settings.active_model_prefix):
    X = get_dummy_input()

    # cold start
    def cold():
        result = get_model(model)
        result.model.predict(X)

    cold_time = measure(cold)

    # warm start
    result = get_model(model)

    # warmup (CPU/cache stabilization)
    for _ in range(warmup_runs):
        result.model.predict(X)

    warm_times = []
    for _ in range(n_runs):
        t = measure(lambda: result.model.predict(X))
        warm_times.append(t)

    return {
        "model": result.model.named_steps["model"].__class__.__name__,
        "cold_start_ms": round(cold_time, 2),
        "warm_avg_ms": round(statistics.mean(warm_times), 2),
        "warm_min_ms": round(min(warm_times), 2),
        "warm_max_ms": round(max(warm_times), 2),
        "warm_std_ms": round(statistics.stdev(warm_times), 2),
        "runs": n_runs,
    }
