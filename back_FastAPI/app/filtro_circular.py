import pycuda
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .utils import cargarImagen
from pydantic import BaseModel
import time


class filtro_circular_Params(BaseModel):
    path_file: str


def creaciónMODCUDACircular():
    modCircular = SourceModule("""
    __global__ void devicecircularmask(const unsigned char* input, const unsigned char* mask, unsigned char* output, int width, int height, int channels) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = (y * width + x) * channels;

        if (x < width && y < height) {
            for (int c = 0; c < channels; ++c) {
                output[idx + c] = input[idx + c] * mask[idx + c];
            }
        }
    }
    """)
    return modCircular


def crear_mascara_circularRGB(alto, ancho, canales, radio):
    mascara = np.zeros((alto, ancho, canales), dtype=np.uint8)
    centro_y = alto // 2
    centro_x = ancho // 2

    for y in range(alto):
        for x in range(ancho):
            for z in range(canales):
                if (y - centro_y) ** 2 + (x - centro_x) ** 2 <= radio ** 2:
                    mascara[y, x, z] = 1
    return mascara


def filtroCircular(path, BloqueX, BloqueY):

    mod = creaciónMODCUDACircular()
    #imagen = cargarImagen(path)
    imagen = cv2.imread(path)
    alto, ancho, canales = imagen.shape
    radio = min(alto, ancho) // 3
    mascara_circular = crear_mascara_circularRGB(alto, ancho, canales, radio)
    imagen_flat = imagen.flatten()
    mascara_flat = mascara_circular.flatten()

    blocks = (BloqueX, BloqueY, 1)
    num_grids = ((ancho + blocks[0] - 1) // blocks[0], (alto + blocks[1] - 1) // blocks[1], 1)

    d_input = cuda.mem_alloc(imagen_flat.nbytes)
    d_mask = cuda.mem_alloc(mascara_flat.nbytes)
    d_output = cuda.mem_alloc(imagen_flat.nbytes)

    cuda.memcpy_htod(d_input, imagen_flat)
    cuda.memcpy_htod(d_mask, mascara_flat)

    func = mod.get_function("devicecircularmask")
    startGPU = time.time()
    func(d_input, d_mask, d_output, np.int32(ancho), np.int32(alto), np.int32(canales), block=blocks, grid=num_grids)
    cuda.Context.synchronize()
    endGPU = time.time()

    tiempo = endGPU - startGPU

    outputImageGPU = np.empty_like(imagen_flat)
    cuda.memcpy_dtoh(outputImageGPU, d_output)
    outputImageGPU = outputImageGPU.reshape((alto, ancho, canales))

    return outputImageGPU, tiempo
