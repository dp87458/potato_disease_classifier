from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import keras

app = FastAPI()

# CORS for frontend access
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the SavedModel as an inference-only layer
MODEL = keras.layers.TFSMLayer("C:/Users/User/potato_disease_model/saved_models/1", call_endpoint="serving_default")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
INPUT_SIZE = (256, 256)

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(INPUT_SIZE)
    image = np.array(image).astype(np.float32) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL(img_batch)

    # Handle prediction dict (Keras 3 + SavedModel often returns dict)
    if isinstance(predictions, dict):
        pred = list(predictions.values())[0][0]
    else:
        pred = predictions[0]

    predicted_class = CLASS_NAMES[np.argmax(pred)]
    confidence = float(np.max(pred))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


