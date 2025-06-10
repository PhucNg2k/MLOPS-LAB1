# 🚀 MLOps Pipeline: Training, Serving, and Monitoring

<div align="center">
  <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 20px;">
    <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="40"/>
    <img src="https://raw.githubusercontent.com/Lightning-AI/lightning/master/docs/source-pytorch/_static/images/logo.png" alt="PyTorch Lightning" height="40"/>
    <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" alt="Optuna" height="40"/>
  </div>
  
  <div style="margin-bottom: 20px;">
    <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
    <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus"/>
    <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white" alt="Grafana"/>
  </div>
</div>

## 📊 Project Overview

A complete MLOps pipeline combining:
- **Model Training**: Neural network training for image classification
- **Model Serving**: FastAPI service for model inference
- **Monitoring & Logging**: Comprehensive system monitoring and logging

## 🏗️ Architecture Components

### 1. Model Training Pipeline
- **PyTorch Lightning**: Streamlined training pipeline with clean code organization
- **Optuna**: Automated hyperparameter optimization with multiple trials
- **MLflow**: Experiment tracking, metrics logging, and model registry
- **Dataset**: FashionMNIST for image classification

#### MLflow Experiment Tracking
Each training experiment is tracked with detailed metrics:
- **Training Metrics**:
  - Loss scores
  - Validation accuracy
  - Test accuracy
- **Model Parameters**:
  - Output dims
  - Number of layers
  - Dropout rates

### 2. Model Serving API
- **FastAPI**: High-performance API for model inference
- **Docker**: Containerized deployment
- **Async Processing**: Efficient request handling
- **Input Validation**: Robust error handling

### 3. Monitoring & Logging
- **Prometheus**:
  - Resource metrics collection (CPU, Memory, GPU)
  - Server performance monitoring
  - API request metrics
  - Health checks for all services
- **Grafana**:
  - Custom dashboards for:
    - Training server metrics
    - API server performance
    - NVIDIA GPU utilization
- **Python Logging**:
  - Structured logs for all services
  - Log rotation and retention
  - Logs stored in 'Logs' directory

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (optional, for GPU training)
- NVIDIA Container Toolkit (for GPU support)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/PhucNg2k/MLOPS-LAB1.git
   cd MLOPS-LAB1
   ```

2. Start all services with a single command:
   ```bash
   # build services
   docker compose build

   # start project
   docker compose up -d
   ```

This will launch:
- Training pipeline with MLflow UI
- FastAPI service (after training completes)
- Prometheus monitoring
- Grafana dashboards
- Logging service

### Accessing Services

- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **FastAPI Swagger**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📊 Monitoring & Logging

### Prometheus Metrics
- Server resource utilization
- API request counts and latencies
- GPU metrics (NVIDIA GPU)
- Custom metrics for model inference

### Grafana Dashboards
- Training server Dashboard
    - CPU usage
    - GPU usage (NVIDIA)
    - RAM usage
    - Disk space, disk IO
    - Network IO 
- API Performance Dashboard
  - Request rates
  - Response times
  - Error rates
  - Inference speed (CPU time và GPU time)
  - Confidence score
  - Total requests count
- GPU Dashboard
  - GPU utilization
  - Memory usage
  - Temperature

### Logging
All logs are stored in the `Logs` directory:
- `server_log.log`: Training server logs
- `api_log.log`: API server logs
- `system_log.log`: Critical system issues logs
- `app_combined.log`": Combining training server and api logs.

## 🔄 Pipeline Flow

1. **Training Phase**:
   - Hyperparameter optimization with Optuna
   - Multiple training trials
   - Metrics logged to MLflow
   - Best model saved automatically

2. **Serving Phase**:
   - Best model loaded from 'mlartifacts' into FastAPI service
   - Ready for inference requests
   - Performance monitored by Prometheus

3. **Monitoring**:
   - Continuous metric collection
   - Real-time dashboard updates
   - Health checks and alerts

## 📝 Notes

- Training must complete before API service starts
- GPU metrics available only with NVIDIA GPUs
- Logs are rotated daily with 30-day retention
- Custom Grafana dashboards are provisioned automatically

## 🎥 Project Development Videos

### Lab Progress Demonstrations

1. **Lab 1: Model Training Pipeline** 
   - Implementation of PyTorch Lightning + Optuna + MLflow
   - [![Lab 1 Demo](https://img.shields.io/badge/Watch_Demo-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=mela8dFpKq0)

2. **Lab 2: FastAPI Model Serving**
   - Deployment of model inference API
   - [![Lab 2 Demo](https://img.shields.io/badge/Watch_Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/6A9RmyL02_k)

3. **Lab 3: Monitoring & Logging**
   - Integration of Prometheus, Grafana, and logging system
   - [![Lab 3 Demo](https://img.shields.io/badge/Watch_Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/E-gGJO-gUTs)

Each video demonstrates the key features and functionality implemented in that development phase.

## 👤 Author

| Full Name           | Student ID |
|---------------------|-----------|
| Nguyen Thuong Phuc  | 22521134  |