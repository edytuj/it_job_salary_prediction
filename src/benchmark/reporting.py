from datetime import datetime

import pandas as pd

from utils.paths import BENCHMARK_FILE


def save_results(results, output_file=BENCHMARK_FILE):
    results["timestamp"] = datetime.now().isoformat()

    df = pd.DataFrame([results])

    if output_file.exists():
        df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


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
