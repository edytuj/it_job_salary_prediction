from prometheus_client import Counter, Histogram, REGISTRY


def metric_exists(name: str) -> bool:
    return name in REGISTRY._names_to_collectors


if metric_exists("salary_prediction_requests"):
    REQUEST_COUNT = REGISTRY._names_to_collectors["salary_prediction_requests"]
else:
    REQUEST_COUNT = Counter(
        "salary_prediction_requests",
        "Total number of API requests",
    )


if metric_exists("salary_prediction_request_latency_seconds"):
    REQUEST_LATENCY = REGISTRY._names_to_collectors[
        "salary_prediction_request_latency_seconds"
    ]
else:
    REQUEST_LATENCY = Histogram(
        "salary_prediction_request_latency_seconds",
        "API request latency in seconds",
    )


if metric_exists("salary_prediction_inference_latency_seconds"):
    PREDICTION_LATENCY = REGISTRY._names_to_collectors[
        "salary_prediction_inference_latency_seconds"
    ]
else:
    PREDICTION_LATENCY = Histogram(
        "salary_prediction_inference_latency_seconds",
        "Prediction latency in seconds",
    )
