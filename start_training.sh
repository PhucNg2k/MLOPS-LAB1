#!/bin/bash

# Exit on any error
set -e

echo "Starting training service..."

# Function to wait for service with timeout
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local sleep_time=$3
    local max_attempts=$4
    local attempt=1

    echo "Waiting for $service_name to be ready..."
    while ! curl -s "$health_url" > /dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo "$service_name not ready after $max_attempts attempts - exiting"
            exit 1
        fi
        echo "$service_name is not ready - sleeping ${sleep_time}s (attempt $attempt/$max_attempts)"
        sleep $sleep_time
        attempt=$((attempt + 1))
    done
    echo "$service_name is ready!"
}

# Wait for MLflow (max 2 minutes: 24 attempts * 5s = 120s)
wait_for_service "MLflow" "http://mlflow:5000/health" 5 24

# Start the metrics server
echo "Starting metrics server..."
python -m uvicorn metrics:app --host 0.0.0.0 --port 8001 &
METRICS_PID=$!

# Wait for metrics server (max 30 seconds: 15 attempts * 2s = 30s)
wait_for_service "Metrics server" "http://localhost:8001/health" 2 15

# Start training
echo "Starting training process..."
if ! python pl_optuna.py; then
    echo "Training failed!"
    kill $METRICS_PID
    exit 1
fi

echo "Training completed successfully!"
kill $METRICS_PID
exit 0 