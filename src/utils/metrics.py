from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("requests_total", "Total number of API requests")

REQUEST_LATENCY = Histogram("request_latency_seconds", "API request latency in seconds")

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
