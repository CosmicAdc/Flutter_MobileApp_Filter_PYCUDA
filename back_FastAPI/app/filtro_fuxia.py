import pycuda
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .utils import cargarImagen
from pydantic import BaseModel
import time


class filtro_turquesa_Params(BaseModel):
    
    path_file: str


def creaciónMODCUDA():
    # Kernel CUDA
    mod = SourceModule("""
    __global__ void devicePixeled(int *input, int size, int *output) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < size / 3) {
            int pixelIndex = tid * 3;  // Indexación para cada componente RGB del píxel
            int val = input[pixelIndex] * 2;
            output[pixelIndex] = (val > 255) ? 255 : val;
            output[pixelIndex + 1] = 0;
            output[pixelIndex + 2] = input[pixelIndex] / 2;
        }
    }
    """)
    return mod


def filtroTurquesa(path, BloqueX, BloqueY):
    # Leer la imagen y convertirla en un array
    img = cv2.imread(path)
    imgarray = img.astype(np.int32)
    alto, ancho, canales = imgarray.shape

    # Calcular los tamaños de bloque y grid
    N = ancho * alto * canales
    block_size = 1024
    grid_size = (N // block_size) + 1

    # Aplanar la imagen para procesar en GPU
    imagen = imgarray.ravel()
    result_gpu = np.empty_like(imagen)

    # Crear el módulo CUDA
    mod = creaciónMODCUDA()
    func = mod.get_function("devicePixeled")

    # Reservar memoria en la GPU
    d_input = cuda.mem_alloc(imagen.nbytes)
    d_output = cuda.mem_alloc(result_gpu.nbytes)

    # Copiar la imagen de entrada a la memoria de la GPU
    cuda.memcpy_htod(d_input, imagen)

    # Ejecutar el kernel CUDA
    startGPU = time.time()
    func(d_input, np.int32(N), d_output, block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.Context.synchronize()
    endGPU = time.time()

    # Copiar el resultado de vuelta a la memoria del host
    cuda.memcpy_dtoh(result_gpu, d_output)

    # Reshape el resultado a la forma original de la imagen
    output_image = result_gpu.reshape((alto, ancho, canales))

    # Tiempo de ejecución en GPU
    tiempo = endGPU - startGPU

    return output_image, tiempo
