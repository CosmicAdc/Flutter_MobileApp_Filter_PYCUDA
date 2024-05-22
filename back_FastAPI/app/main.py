import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException,Request
import cv2
import numpy as np
import uuid
from .filtro_circular import filtroCircular, filtro_circular_Params
from .filtro_fuxia import filtroTurquesa, filtro_turquesa_Params
from .filtro_logo_ups import filtroLogo, filtro_logo_Params
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .analitics import auth

## EN EL POETRY HAY QUE HACER UN PIP INSTALL sqlalchemy y un pip install psycopg2 ya que debe hacerse desde el SO

app = FastAPI(static_directory="app/static")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

SAVE_PATH_ORIGINAL = "app/static/originales/"
SAVE_PATH_CIRCULO = "app/static/circulo/"
SAVE_PATH_MAREA = "app/static/marea/"
SAVE_PATH_UPS = "app/static/ups/"

##Subir imagen al servidor
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Leer la imagen en color

        filename = os.path.splitext(file.filename)[0]

        # Guardar la imagen en color (tres canales)
        cv2.imwrite(f"{SAVE_PATH_ORIGINAL}{filename}.jpg", img)
        name_File= f"{SAVE_PATH_ORIGINAL}{filename}.jpg"
        print(f"File '{file.filename}' guardado")
    return {"message": "Subida exitosamente", "name_file":name_File}


@app.post("/filtroCirculo/")
async def filtro_sobel(params: filtro_circular_Params):
    bloques_x = 32
    bloques_y = 32
    path_file = params.path_file
    nombre = path_file[22:]
    path=str(path_file)
    imagenFinal, tiempo= filtroCircular(path,bloques_x,bloques_y)
    path_final=SAVE_PATH_CIRCULO+'Circulo-'+str(nombre)
    print(path_final)
    cv2.imwrite(path_final, imagenFinal)
    return {"ruta_imagen": path_final, "tiempo": tiempo}

@app.post("/filtroUPS/")
async def filtro_gauss(params: filtro_logo_Params):
    bloques_x = 32
    bloques_y = 32
    path_file = params.path_file
    nombre = path_file[22:]
    path=str(path_file)
    path_final=SAVE_PATH_UPS+'UPS-'+str(nombre)

    imagenFinal, tiempo = filtroLogo(path,bloques_x,bloques_y)

    cv2.imwrite(path_final, imagenFinal)

    return {"ruta_imagen": path_final, "tiempo": tiempo}

@app.post("/filtroTurquesa/")
async def filtro_gauss(params: filtro_turquesa_Params):
    bloques_x = 32
    bloques_y = 32
    path_file = params.path_file
    nombre = path_file[22:]
    path=str(path_file)
    path_final=SAVE_PATH_MAREA+'Turquesa-'+str(nombre)

    imagenFinal, tiempo = filtroTurquesa(path,bloques_x,bloques_y)

    cv2.imwrite(path_final, imagenFinal)

    return {"ruta_imagen": path_final, "tiempo": tiempo}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)