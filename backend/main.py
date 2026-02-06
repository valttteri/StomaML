from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import logging
from contextlib import asynccontextmanager
from utils.inference import batch_inference
from utils.model_utils import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_manager.load_model("models/weights.pt")
        logger.info("Model loaded")
    except Exception:
        logger.exception("Failed to load model")
        raise
    yield


app = FastAPI(title="Stoma Segmentation API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Stoma Segmentation API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_manager.is_loaded()}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...),conf: float = Query(0.25, ge=0.0, le=1.0)):
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images_data = []
    for file in files:
        contents = await file.read()
        if not contents:
            continue
        images_data.append((file.filename, contents))

    if not images_data:
        raise HTTPException(status_code=400, detail="All uploaded files were empty")

    model = model_manager.get_model()
    results = batch_inference(model, images_data, conf_threshold=conf)

    return JSONResponse(content={"total_images": len(images_data), "results": results})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) #change this for test/prod, no need for reload 
