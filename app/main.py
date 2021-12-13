import cv2
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import requests
from tensorflow import keras, reshape
from tensorflow_hub import KerasLayer
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"mensaje": "Página de inicio de la API del reconocedor de dibujos"}


@app.post('/predict')
def predict(imagen: UploadFile = File(...)):
    modelo = importar_modelo("../model/model.h5")
    return categorizar(modelo, imagen)


def importar_modelo(ruta):
    return keras.models.load_model(ruta, custom_objects={'KerasLayer': KerasLayer})


def categorizar(modelo, imagen):
    respuesta = requests.get(imagen)
    img = Image.open(BytesIO(respuesta.content))

    # Elimina el canal alpha, solo deja RGB
    img = img.convert('RGB')
    img = np.array(img).astype(float) / 255

    print('Tamaño original:', img.shape)

    # Cambia el tamaño de la imagen
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    print('Tamaño nuevo:', img.shape)

    prediccion = modelo.predict(reshape(img, [-1, 224, 224, 3]))
    resultado = np.argmax(prediccion[0], axis=-1)

    categorias = ["data_table", "image", "radio_button_checked", "slider", "text_area"]
    print("Es " + categorias[resultado])

    return resultado
