import pycuda
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .utils import cargarImagen
from pydantic import BaseModel
import time


class filtro_logo_Params(BaseModel):
    
    path_file: str


def creaciónMODCUDA():
    # Kernel CUDA
    mod = SourceModule("""
    __global__ void devicelogo(int *input, int *mask, int size, int *output) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < size / 3) {
            int pixelIndex = tid * 3;  // Indexación para cada componente RGB del píxel
            if (mask[pixelIndex] == 255 && mask[pixelIndex +1] == 255 && mask[pixelIndex +2] == 255) {
                output[pixelIndex] = input[pixelIndex];
                output[pixelIndex + 1] = input[pixelIndex + 1];
                output[pixelIndex + 2] = input[pixelIndex + 2];
            } else {
                output[pixelIndex] = mask[pixelIndex];
                output[pixelIndex + 1] = mask[pixelIndex + 1];
                output[pixelIndex + 2] = mask[pixelIndex + 2];
            }
        }
    }
    """)
    return mod


def filtroLogo(path_img, BloqueX, BloqueY):
    # Leer la imagen y la máscara, y convertirlas en arrays
    img = cv2.imread(path_img)
    mask = cv2.imread("app/static/originales/marco.png")
    
    imgarray = img.astype(np.int32)
    alto, ancho, canales = imgarray.shape

    # Redimensionar la máscara para que coincida con el tamaño de la imagen
    resized_mask = cv2.resize(mask, (ancho, alto))

    # Calcular los tamaños de bloque y grid
    N = ancho * alto * canales
    block_size = 1024
    grid_size = (N // block_size) + 1

    # Aplanar la imagen y la máscara para procesar en GPU
    imagen = imgarray.ravel()
    mascara = resized_mask.astype(np.int32).ravel()
    result_gpu = np.empty_like(imagen)

    # Crear el módulo CUDA
    mod = creaciónMODCUDA()
    func = mod.get_function("devicelogo")

    # Reservar memoria en la GPU
    d_input = cuda.mem_alloc(imagen.nbytes)
    d_mask = cuda.mem_alloc(mascara.nbytes)
    d_output = cuda.mem_alloc(result_gpu.nbytes)

    # Copiar la imagen y la máscara de entrada a la memoria de la GPU
    cuda.memcpy_htod(d_input, imagen)
    cuda.memcpy_htod(d_mask, mascara)

    # Ejecutar el kernel CUDA
    startGPU = time.time()
    func(d_input, d_mask, np.int32(N), d_output, block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.Context.synchronize()
    endGPU = time.time()

    # Copiar el resultado de vuelta a la memoria del host
    cuda.memcpy_dtoh(result_gpu, d_output)

    # Reshape el resultado a la forma original de la imagen
    output_image = result_gpu.reshape((alto, ancho, canales))

    # Tiempo de ejecución en GPU
    tiempo = endGPU - startGPU

    return output_image, tiempo

