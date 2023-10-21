# Fishsnap Backend

Fishsnap Backend is a Python script that uses a custom trained Keras-based Imagenetv3 model to classify fish species in an image. This script is intended to be used as a backend for a larger fish image classification project.

## Usage

To use Fishsnap Backend, you must have Python 3.6 or higher, TensorFlow 2.0 or higher, NumPy, and Pillow installed. Use the following command to install dependancies:

```bash
pip install -r requirements.txt
```

or use an environment using `conda` or `venv`

This script is a backend for a fish image classification project. It takes in an image file and returns a prediction of the fish species in the image. The script uses a custom trained Keras-based Imagenetv3 model to make the prediction.

### Run the app:

```bash
python app.py
```

## FastAPI-based Web Backend

Fishsnap Backend also includes a FastAPI-based web backend that takes an image file URL as input and returns the predicted fish species name. To use the web backend, you must have Uvicorn and Gunicorn installed. Once you have these dependencies installed, you can start the web backend with the following command:
