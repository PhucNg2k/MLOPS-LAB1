# Serve API 🚀

A **FastAPI-based** image prediction service running in Docker.

<div align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</div>


## 📋 Overview

This API allows you to upload images and receive predictions through a simple interface.

## 🔧 Prerequisites

- Docker
- Docker Compose

## 🚀 Getting Started

### Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/PhucNg2k/MLOPS-LAB1.git
   cd MLOPS-LAB1/ServeAPI
   ```

2. **Build image and run containers**
   ```bash
   docker compose up --build
   ```

3. **Verify containers are running**
   ```bash
   docker ps
   ```

## 🌐 Usage

- **API endpoint**: [localhost:8000](http://localhost:8000)
- **API documentation**: [localhost:8000/docs](http://localhost:8000/docs)

### Making Predictions

1. Navigate to root page [localhost:8000](http://localhost:8000).
2. Upload your image using the provided interface
3. Click the "Submit" button to receive predictions

## 🛑 Stopping the Service

To stop and remove containers:

```bash
docker compose down
```

