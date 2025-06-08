#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting API Traffic Simulation${NC}"

# Function to send normal prediction requests
simulate_normal_traffic() {
    echo -e "${GREEN}Simulating normal prediction traffic...${NC}"
    for i in {1..50}; do
        # Send prediction request with sample image
        curl -s -X POST -F "file=@sample.jpg" http://localhost:8000/predict > /dev/null
        echo -e "${GREEN}Sent normal request $i${NC}"
        # Random sleep between 0.1 to 0.5 seconds
        sleep 0.$(( RANDOM % 5 + 1 ))
    done
}

# Function to simulate error cases
simulate_error_traffic() {
    echo -e "${RED}Simulating error cases...${NC}"
    
    # Case 1: Invalid file type
    echo "Testing invalid file type..."
    curl -s -X POST -F "file=@requirements.txt" http://localhost:8000/predict
    sleep 1

    # Case 2: Missing file
    echo "Testing missing file..."
    curl -s -X POST http://localhost:8000/predict
    sleep 1

    # Case 3: Invalid endpoint
    echo "Testing invalid endpoint..."
    curl -s http://localhost:8000/invalid_endpoint
    sleep 1

    # Case 4: Invalid HTTP method
    echo "Testing invalid HTTP method..."
    curl -s -X PUT http://localhost:8000/predict
    sleep 1
}

# Function to check logs
check_logs() {
    echo -e "${GREEN}Checking API logs...${NC}"
    tail -n 20 ../Logs/api_log.log
    
    echo -e "${GREEN}Checking System logs...${NC}"
    tail -n 20 ../Logs/system_log.log
}

# Main test sequence
echo "=== Starting API Traffic Test ==="
echo "1. First wave of normal traffic"
simulate_normal_traffic

echo "2. Simulating error cases"
simulate_error_traffic

echo "3. Second wave of normal traffic"
simulate_normal_traffic

echo "4. Checking logs"
check_logs

echo -e "${GREEN}Test completed. Please check Grafana dashboard for metrics visualization${NC}"
echo "Dashboard URL: http://localhost:3000" 