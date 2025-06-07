from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app
import time
import mlflow

from PIL import Image
import uvicorn
import os
import sys
import time
import torch

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils import (
    normalize_path,
    find_mlrun,
    get_best_modelFile,
    load_model,
    preprocess_image,
    inference_model
)

# Import logging and monitoring
from LOGGING_SERVICE.logger import get_logger, shutdown_logging
from MONITORING_SERVICE.monitoring import api_metrics_manager as metrics_manager

# Initialize API loggers
app_logger = get_logger('app', 'api')
sys_logger = get_logger('syslog', 'api')

# Verify loggers are initialized
if not all([app_logger, sys_logger]):
    raise RuntimeError("Failed to initialize API loggers")

model = None  # Global model reference

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    max_retries = 300  # Wait up to 5 minutes (30 * 10 seconds)
    retry_count = 0
    
    try:
        app_logger.info("Starting API server...")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        while retry_count < max_retries:
            try:
                # Try to get the best model directly from MLflow
                model_path = get_best_modelFile(None, None)  # We don't need these parameters anymore

                if model_path is not None:
                    sys_logger.info(f"Found best model at: {model_path}")
                    # Load model
                    model = load_model(model_path)
                    if model is not None:
                        sys_logger.info("Model loaded successfully")
                        break
                    else:
                        sys_logger.warning("Failed to load model, retrying...")
                else:
                    sys_logger.info("Waiting for model to be available...")
                time.sleep(10)  # Wait 10 seconds before retrying
                retry_count += 1
            except Exception as e:
                sys_logger.warning(f"Attempt {retry_count + 1}: {str(e)}")
                time.sleep(10)
                retry_count += 1
        
        if model is None:
            sys_logger.error("Failed to load model after maximum retries")
            raise RuntimeError("No model available after waiting")
            
    except Exception as e:
        sys_logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

    yield  # The app runs here

    sys_logger.info("Shutting down API server...")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    metrics_manager.record_request_metrics(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )
    
    return response

@app.get("/", response_class=HTMLResponse)
async def root():
    app_logger.debug("Serving index page")
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    app_logger.info(f"Received prediction request for file: {file.filename}")
    
    if not file.content_type.startswith("image/"):
        app_logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        image = Image.open(file.file)
        app_logger.debug("Image loaded successfully")
        
        # Start timing
        cpu_start = time.time()
        
        image_tensor = preprocess_image(image)
        app_logger.debug("Image preprocessed")
        
        # Record GPU start time and sync GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_start = time.time()
        
        # Run inference
        prediction, confidence = inference_model(image_tensor, model, return_confidence=True)
        
        # Get GPU time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start
        else:
            gpu_time = 0
            
        # Get total CPU time
        cpu_time = time.time() - cpu_start
        
        # Update metrics
        metrics_manager.update_gpu_metrics()
        metrics_manager.record_inference_metrics(gpu_time, cpu_time, confidence)
        
        app_logger.info(f"Prediction completed: {prediction}, confidence: {confidence}")
        
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "metrics": {
                "gpu_time": gpu_time,
                "cpu_time": cpu_time
            }
        })
    except Exception as e:
        app_logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    try:
        app_logger.info("Starting FastAPI server")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # Allow external connections
            port=8000,
            reload=True      # Enable auto-reload on code changes
        )
    except Exception as e:
        sys_logger.error(f"Server failed to start: {str(e)}", exc_info=True)
    finally:
        shutdown_logging('api')
