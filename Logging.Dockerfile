FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy logging service code
COPY LOGGING_SERVICE /app/LOGGING_SERVICE
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create log directory
RUN mkdir -p Logs

# Copy and run the logging service
CMD ["python", "-m", "LOGGING_SERVICE.logger"] 