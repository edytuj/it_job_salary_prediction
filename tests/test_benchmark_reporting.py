import pandas as pd

from src.benchmark.reporting import save_results


def test_save_results_creates_csv(tmp_path):
    benchmark_file = tmp_path / "benchmark.csv"

    results = {
        "model": "HistGradientBoostingRegressor",
        "cold_start_ms": 500.0,
        "warm_avg_ms": 5.0,
        "warm_min_ms": 4.5,
        "warm_max_ms": 6.0,
        "warm_std_ms": 0.5,
        "runs": 20,
    }

    save_results(results, benchmark_file)

    assert benchmark_file.exists()

    df = pd.read_csv(benchmark_file)

    assert len(df) == 1
    assert df.loc[0, "model"] == "HistGradientBoostingRegressor"
    assert "timestamp" in df.columns


def test_save_results_appends_rows(tmp_path):
    benchmark_file = tmp_path / "benchmark.csv"

    results = {
        "model": "Ridge",
        "cold_start_ms": 100.0,
        "warm_avg_ms": 1.0,
        "warm_min_ms": 0.8,
        "warm_max_ms": 1.2,
        "warm_std_ms": 0.1,
        "runs": 10,
    }

    save_results(results, benchmark_file)
    save_results(results, benchmark_file)

    df = pd.read_csv(benchmark_file)

    assert len(df) == 2
