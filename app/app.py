import io

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from inference.predictor import Predictor
from loguru import logger
from models.detector import AerialDetector
from PIL import Image

app = FastAPI(title="Aerial Object Detection API")

with open("app/config/config.yaml") as f:
    config = yaml.safe_load(f)

model = AerialDetector(len(config["data"]["classes"]))
try:
    model.load_state_dict(
        torch.load(
            "/home/sadikshya/Downloads/AngelSwing/ObjectDetection/checkpoints/model_epoch_2.pth"
        )
    )
    predictor = Predictor(model, config)
except Exception as e:
    logger.error(f"[ERROR] Error loading model: {e}")
    raise


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in aerial image

    Returns:
    {
        "prediction": [
            {
                "label": str,        # One of: "car", "house", "road", "swimming pool", "tree", "yard"
                "confidence": float, # Value between 0.0 and 1.0
                "bbox": list        # [x_min, y_min, x_max, y_max]
            },
            ...
        ]
    }
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400, detail="Only JPEG and PNG images are supported"
            )

        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        predictions = predictor.predict(image)
        response = {"prediction": predictions}

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/healthz/")
async def check():
    return {"message": "Welcome to aerial object detection project."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
