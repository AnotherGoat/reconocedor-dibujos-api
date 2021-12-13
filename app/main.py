from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
from urllib import request

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


class Imagen(BaseModel):
    uri: str


@app.post('/predict')
def predict(imagen: Imagen):
    modelo = importar_modelo("model/model.h5")
    img = leer_imagen(imagen.uri)
    return evaluar(modelo, img)


def importar_modelo(ruta):
    return tf.keras.models.load_model(ruta, custom_objects={'KerasLayer': hub.KerasLayer})


def leer_imagen(data_uri):
    respuesta = request.urlopen(data_uri).read()
    return Image.open(BytesIO(respuesta))


def evaluar(modelo, img):
    # Elimina el canal alpha, solo deja RGB
    img = img.convert('RGB')
    img = np.array(img).astype(float) / 255

    prediccion = modelo.predict(tf.reshape(img, [-1, 224, 224, 3]))
    resultado = np.argmax(prediccion[0], axis=-1)

    categorias = ["data_table", "image", "radio_button_checked", "slider", "text_area"]
    print("Es " + categorias[resultado])

    return categorias[resultado]
