FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories first and set proper permissions
RUN mkdir -p Logs mlruns mlartifacts data checkpoints && \
    chmod -R 777 Logs mlruns mlartifacts data checkpoints

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyyaml

# Copy necessary files
COPY pl_optuna.py .
COPY metrics.py .
COPY LOGGING_SERVICE /app/LOGGING_SERVICE
COPY MONITORING_SERVICE /app/MONITORING_SERVICE
COPY model /app/model
COPY start_training.sh .

# Fix line endings and make executable
RUN dos2unix start_training.sh && chmod +x start_training.sh

# Command to run the startup script
CMD ["./start_training.sh"] 