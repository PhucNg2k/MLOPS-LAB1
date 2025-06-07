FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ServeAPI/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories first
RUN mkdir -p Logs mlruns mlartifacts static

# Copy app files
COPY ServeAPI/main.py .
COPY ServeAPI/utils.py .
COPY ServeAPI/static/* ./static/
COPY LOGGING_SERVICE /app/LOGGING_SERVICE
COPY MONITORING_SERVICE /app/MONITORING_SERVICE
COPY model /app/model

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]