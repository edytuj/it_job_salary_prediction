import time
import statistics
import pandas as pd

from api.model_loader import get_model


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


def benchmark(n_runs=20, warmup_runs=5):
    X = get_dummy_input()

    # cold start
    def cold():
        model = get_model()
        model.predict(X)

    cold_time = measure(cold)

    # warm start
    model = get_model()

    # warmup (CPU/cache stabilization)
    for _ in range(warmup_runs):
        model.predict(X)

    warm_times = []
    for _ in range(n_runs):
        t = measure(lambda: model.predict(X))
        warm_times.append(t)

    return {
        "cold_start_ms": round(cold_time, 2),
        "warm_avg_ms": round(statistics.mean(warm_times), 2),
        "warm_min_ms": round(min(warm_times), 2),
        "warm_max_ms": round(max(warm_times), 2),
        "warm_std_ms": round(statistics.stdev(warm_times), 2),
        "runs": n_runs,
    }


if __name__ == "__main__":
    results = benchmark()

    print("\nBenchmark results:")
    for key, value in results.items():
        print(f"{key}: {value}")
