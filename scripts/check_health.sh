#!/bin/bash

URL="http://localhost:8000/health"

response=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

if [ "$response" -eq 200 ]; then
    echo "API health check passed"
    exit 0
else
    echo "API health check failed with status $response"
    exit 1
fi
