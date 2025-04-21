from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI(title="CNN Image Classification API")

# Load model and class names
model = load_model("model.h5")
IMG_SIZE = (224, 224)

with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

# image preprocessing
def preprocess(file: UploadFile):
    img_bytes = file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img_array = preprocess(file)
        prediction = model.predict(img_array)[0]
        class_idx = np.argmax(prediction)
        class_name = class_names[class_idx]
        confidence = float(prediction[class_idx])
        return {
            "predicted_class": class_name,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
