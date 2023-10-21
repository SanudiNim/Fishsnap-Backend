from io import BytesIO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
import requests
import uvicorn

# Set CUDA_VISIBLE_DEVICES to an empty string to force CPU usage

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

import numpy as np
import uuid

save_directory = "path/to/save/images"
app = FastAPI()

# Load your pre-trained model
model = load_model("model/fish_model.h5")

def download_image(image_url: str) -> BytesIO:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Read image content and store it in BytesIO object
        image_bytesio = BytesIO(response.content)
        return image_bytesio

    except Exception as e:
        print(f"Error downloading image: {e}")
        # Handle the error according to your requirements (e.g., return an error response)



@app.post("/predict/")
def predict_image(image_url: str):
    image_bytesio = download_image(image_url)
    img = image.load_img(image_bytesio, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess the image
    predictions = model.predict(x)

    predicted_class_index = np.argmax(predictions, axis=1)[0]

    class_labels = [
        "Bangus",
        "Big Head Carp",
        "Black Spotted Barb",
        "CatFish",
        "Climbing Perch",
        "FourFinger Thredfin",
        "FreshWater Eel",
        "Glass Perchlet",
        "Goby",
        "Gold Fish",
        "Gourami",
        "Grass Crap",
        "Green Spotted Puffer",
        "Indian Crap",
        "Indo-Pacific Tarpon",
        "Jaguar Gapote",
        "Janitor Fish",
        "Knife Fish",
        "Long- Snouted PipeFish",
        "Mosquito Fish",
        "MudFish",
        "Mullet",
        "Pangasius",
        "Perch",
        "Scat Fish",
        "Silver Barb",  
        "Silver Carp",
        "Silver Perch",
        "Snakehead",
        "TenPounder",
        "Thilapia",]

    predicted_label = class_labels[predicted_class_index]

    return {"index": predicted_class_index}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
