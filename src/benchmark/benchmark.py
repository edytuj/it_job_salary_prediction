import statistics
import time
from datetime import datetime

import pandas as pd

from config.settings import settings
from model.model_loader import get_model
from utils.paths import BENCHMARK_FILE


def get_dummy_input():
    return pd.DataFrame(
        [
            {
                "title_clean": "python developer",
                "skills_clean": ["python", "aws"],
                "city_clean": "Warszawa",
                "seniority": "mid",
                "skills_count": 2,
            }
        ]
    )


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


def save_results(results):
    results["timestamp"] = datetime.now().isoformat()

    df = pd.DataFrame([results])

    if BENCHMARK_FILE.exists():
        df.to_csv(BENCHMARK_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(BENCHMARK_FILE, index=False)

    print(f"Results saved to {BENCHMARK_FILE}")


def show_results():
    if BENCHMARK_FILE.exists():
        df = pd.read_csv(BENCHMARK_FILE)

        summary = (
            df.groupby("model")["warm_avg_ms"]
            .mean()
            .reset_index()
            .rename(columns={"warm_avg_ms": "avg_latency_ms"})
        )

        summary = summary.sort_values("avg_latency_ms")

        print(" Benchmark summary ".center(46, "-"))
        print(summary.to_string(index=False))
        print(f"\n{'-' * 46}\n")
    else:
        print("No results found.")
