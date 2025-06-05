from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from PIL import Image
import uvicorn
import os
import sys

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

# Import logging service
from LOGGING_SERVICE.logger import LoggingManager

# Initialize logging
logging_manager = LoggingManager()
logging_manager.setup_logging()

# Get app logger
app_logger = logging_manager.get_logger('app')

model = None  # Global model reference

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        app_logger.info("Starting API server...")

        # Locate best model path
        folder_root = normalize_path("mlruns")
        target_run = find_mlrun(folder_root)
        folder_root = normalize_path("mlartifacts")
        model_path = get_best_modelFile(folder_root, target_run)

        if model_path is None:
            app_logger.error("Best model path not found")
            raise RuntimeError("Best model path not found.")
        
        app_logger.info(f"Found best model at: {model_path}")
        
        # Load model
        model = load_model(model_path)
        app_logger.info("Model loaded successfully")
    except Exception as e:
        app_logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

    yield  # The app runs here

    app_logger.info("Shutting down API server...")


app = FastAPI(lifespan=lifespan)

# mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        
        image_tensor = preprocess_image(image)
        app_logger.debug("Image preprocessed")
        
        prediction = inference_model(image_tensor, model, return_label=True)
        app_logger.info(f"Prediction completed: {prediction}")
        
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        app_logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


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
        app_logger.error(f"Server failed to start: {str(e)}", exc_info=True)
    finally:
        logging_manager.shutdown()
