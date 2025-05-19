from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from PIL import Image
import uvicorn
import os

from utils import (
    normalize_path,
    find_mlrun,
    get_best_modelFile,
    load_model,
    preprocess_image,
    inference_model
)


model = None  # Global model reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("🔄 Loading model during startup...")

        # Locate best model path
        folder_root = normalize_path("mlruns")
        target_run = find_mlrun(folder_root)
        folder_root = normalize_path("mlartifacts")
        model_path = get_best_modelFile(folder_root, target_run)

        if model_path is None:
            raise RuntimeError("Best model path not found.")
        
        # Load model
        model = load_model(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    yield  # The app runs here

    print("🧹 Cleanup if necessary...")


app = FastAPI(lifespan=lifespan)

# mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        image = Image.open(file.file)
        image_tensor = preprocess_image(image)
        prediction = inference_model(image_tensor, model, return_label=True)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
