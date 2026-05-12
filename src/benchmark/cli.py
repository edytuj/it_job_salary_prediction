from argparse import ArgumentParser, Namespace

from benchmark.benchmark import (
    run_benchmark,
    save_results,
    show_results,
)
from config.model_types import ModelPrefix


def parse_args() -> Namespace:
    """Parse command-line arguments for benchmark CLI."""

    parser = ArgumentParser(
        description="Benchmark ML model inference performance",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # benchmark run
    run_parser = subparsers.add_parser(
        "run",
        help="Run inference benchmark",
    )

    run_parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=20,
        help="Number of warm inference runs",
    )

    run_parser.add_argument(
        "--model",
        "-m",
        choices=[model.value for model in ModelPrefix],
        default=ModelPrefix.HGB.value,
        help="Model prefix to benchmark",
    )

    # benchmark show
    subparsers.add_parser(
        "show",
        help="Show aggregated benchmark results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "run":
        results = run_benchmark(n_runs=args.runs, model=args.model)

        print("\nBenchmark results:")

        for key, value in results.items():
            print(f"{key}: {value}")

        save_results(results)

    elif args.command == "show":
        show_results()


if __name__ == "__main__":
    main()
