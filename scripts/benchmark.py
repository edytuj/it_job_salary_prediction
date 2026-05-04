import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
import pandas as pd

from api.model_loader import get_model

OUTPUT_FILE = Path("benchmark_results.csv")


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
        "model": model.named_steps["model"].__class__.__name__,
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

    if OUTPUT_FILE.exists():
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)

    print(f"Results saved to {OUTPUT_FILE}")


def print_results():
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print(df.groupby("model")["warm_avg_ms"].mean())
    else:
        print("No results found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py [run|show]")

    elif sys.argv[1] == "run":
        results = benchmark()

        print("\nBenchmark results:")
        for key, value in results.items():
            print(f"{key}: {value}")

        save_results(results)

    elif sys.argv[1] == "show":
        print_results()

    else:
        print("Unknown command")
