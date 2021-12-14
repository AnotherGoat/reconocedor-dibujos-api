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
    img = leer_imagen(imagen.uri)
    modelo = importar_modelo("model/model.h5")
    return evaluar(modelo, img)


def importar_modelo(ruta):
    return tf.keras.models.load_model(ruta, custom_objects={'KerasLayer': hub.KerasLayer})


def leer_imagen(data_uri):
    respuesta = request.urlopen(data_uri).read()
    return Image.open(BytesIO(respuesta))


def evaluar(modelo, img):
    # Pasa la imagen a RGB
    img = imagen_a_rgb(img)

    # Se divide por 255 para que sea la forma de entrada esperada
    img = np.array(img).astype(int) / 255

    # Realiza la predicción
    prediccion = modelo.predict(tf.reshape(img, [-1, 224, 224, 3]))
    del modelo # Libera memoria
    resultado = np.argmax(prediccion[0], axis=-1)

    categorias = ["Checkbox (checked)", "Checkbox (unchecked)", "Data Table", "Dropdown Menu",
                  "Floating Action Button", "Image", "Label", "Radio Button (checked)",
                  "Radio Button (unchecked)", "Slider", "Switch (disabled)", "Switch (enabled)",
                  "Text Area", "Tooltip"]
    print("Es " + categorias[resultado])

    return categorias[resultado]


def imagen_a_rgb(img):
    # Convierte imagen a RGB, tomando en cuenta la transparencia
    img_rgb = Image.new("RGB", img.size, (255, 255, 255))
    img_rgb.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    return img_rgb


def imagen_info(img):
    print("Format: ", img.format)
    print("Mode: ", img.mode)
    print("Size: ", img.size)
    print("Width: ", img.width)
    print("Height: ", img.height)
    print("Image Palette: ", img.palette)
    print("Image Info: ", img.info)
    print("First Pixel: ", img.getpixel((0, 0)))
