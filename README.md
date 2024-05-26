![Logo](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/12a82279-0ac6-45bc-95a6-76ea53c9eaf6)

# Aplicación móvil - Pycuda - Flutter
 Parallel Computing Integrator Project

 ## Carrera: Computación

 ## Asignatura: Computación Paralela

 ## Integrantes:

 Andrés Alba
  
 Christian Buestan

 Santiago Torres



 ## Objetivo alcanzado: 
Desarrollar una aplicación flutter de edición de imágenes por medios de filtros personalizados en base a algoritmos de convolución utilizando PyCUDA, aprovechando la potencia de procesamiento GPU, esta convolución se realiza en una API con comunicación a una Base de datos; la aplicación móvil llama a esta API para mostrar los resultados, esta API es dockerizada y ejecutada.

## Actividades desarrolladas
1. Se programo 3 filtros personalizados. Siendo un filtro respectivo a icono de la UPS. Estos filtros fuero en base a algoritmos utilizando PyCUDA y el procesamiento de la GPU
   1 Filtro del logo de la UPS que se combina con la imagen de forma que quede como una marca de agua.
   1 Filtro circular de radio modificable, enfocando solo el contenido de la imagen dentro del radio.
   1 Filtro fuxia que modifica los pixeles de la imagen para que predomine el canal Rojo, el canal verde se disminuya y que el canal azul se medialize.
3. Se desarrollo una API que use estos algoritmos en llamados REQUEST POST para aplicar uno de los 3 filtros a una imagen dada y que regrese el resultado de este proceso.
   En esta API se llaman archivos PYTHON para cada filtro a aplicar en la imagen, la imagen es cargada una función POST, con ello el usuario llamara la función POST llamando a la imagen y seleccionando el filtro respectivo a aplicar, el resultado es la imagen resultante y el tiempo de ejecución del proceso.
5. Se realizo una comunicación a base de datos POSTGRESQL para hacer 2 procesos, login y register de usuarios, aplicados dentro de la API.
   Este proceso fue mediante CRUD de un usuario y la conexión al DB, donde se puede registrar un nuevo usuario a la base de datos, también con login el usuario iniciara sesión dentro de la aplicación, si es exitoso el sistema devolverá el nombre de usuario y su ID respectivo, estos procesos de Register y login son accedidos por URL HTTP desde la aplicación móvil.
7. Se desarrollo una aplicación móvil usando flutter y comunicaciones por medio de URL de los request de la API, esto para las páginas de login, register y convolución de imágenes por medio de imágenes del dispositivo móvil.
   Se usaron archivos DART para el diseño y funcionalidad de conexión HTTP a las funciones de PYCUDA y sistema de Login/Register de la API, esto además de un diseño intuitivo para el usuario, asi como el manejo de errores de conexión a la API para la garantizar la disponibilidad de la aplicación. Esto permitirá generar la APK de la aplicación para poder instalarla en los sistemas móviles. Las páginas de login y register son la página principal para verificar que el usuario este autorizado, luego con ello se guiara a la página de convoluciones de imágenes donde se capturara la imagen con la cámara del dispositivo, posteriormente se seleccionara el tipo de filtro a aplicar para ser guardado y mostrado en la misma aplicación.
9. Dockerizacion de la API y la base de datos.
   Se creó un Dockerfile para construir la imagen de la API con Python 3.10, CUDA 12.2.2 y las dependencias necesarias. La imagen se basa en nvidia/cuda:12.2.2-devel-ubi8 e incluye la instalación de Poetry para gestionar las dependencias.
   Además, se generó un archivo docker-compose.yml para orquestar la ejecución de dos servicios: la API y la base de datos PostgreSQL.
   El servicio web construye la imagen de la API utilizando el Dockerfile, expone el puerto 8000 y se configura con la URL de la base de datos.
   El servicio db utiliza la imagen oficial de PostgreSQL 13, configura las credenciales, expone el puerto 5432 y define un volumen para persistir los datos.
10. Generación de la APK.

## Resultados obtenidos
Página principal LOGIN de un nuevo usuario:

![Imagen de WhatsApp 2024-05-24 a las 12 18 19_050fb970](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/358f772a-0763-4332-9fa8-b1a9d69be423)

Página de registro de usuarios dentro de la base de datos.

![Imagen de WhatsApp 2024-05-24 a las 12 18 19_74e8bf17](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/a3457e1e-8e60-45ed-b2ca-27a25081312b)

Manejo de errores con respecto a la comunicación con el BackEnd API.

![Imagen de WhatsApp 2024-05-23 a las 04 35 09_1efcbdef](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/3b477b5a-4268-453f-9ef0-4f6fbadd1908)

Generación del DockerFile y Dockerizacion de la API.

![Imagen de WhatsApp 2024-05-24 a las 00 30 21_fcc247fe](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/726118c7-b3d9-4ef9-b0e9-648d4a1bae56)


Uso de la cámara con Flutter para obtener la imagen a procesar.

![Imagen de WhatsApp 2024-05-25 a las 20 33 12_ea5737c8](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/79539900-5924-4e02-947f-a345f1646e1d)

Despliegue de menú de filtros a aplicar a la imagen.

![Imagen de WhatsApp 2024-05-25 a las 20 33 12_a66d0e5b](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/e65a869d-289c-4c30-b2fd-10a8de175836)


Selección de filtro circular a aplicar sobre la imagen tomada.

![Imagen de WhatsApp 2024-05-25 a las 20 33 13_7e64cc97](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/ac13300a-b263-4110-b39b-604b09b07aae)


Selección de filtro fuxia a aplicar en la imagen tomada.

![Imagen de WhatsApp 2024-05-25 a las 20 33 13_306f8520](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/a7f0b91a-daae-43ad-a306-7518f6b5abcb)


Selección de filtro UPS a aplicar en la imagen tomada.

![Imagen de WhatsApp 2024-05-25 a las 20 33 13_c95acb5f](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/e1f77451-e2bc-4704-a492-0ec699168090)


-Cuando se haya definido el filtro a aplicar se selecciona el botón de enviar del menú para guardar el resultado con un nombre de archivo ingresado por el usuario.

![Imagen de WhatsApp 2024-05-25 a las 20 33 13_f4c3f132](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/220d9748-4cff-432b-b08a-32fabe512b2a)


Guardado exitoso de la imagen resultante de aplicar el filtro circular y su visualización en la aplicación móvil.

![Imagen de WhatsApp 2024-05-25 a las 20 33 13_fdabb0a9](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/b507fe6f-f999-4508-8ee3-ec3ab9a04f86)

Dockerizacion del BackEnd fastAPI en base al dockerfile.

![Imagen de WhatsApp 2024-05-26 a las 10 41 00_4aa8d98d](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/d4cc8865-beff-46e3-8560-1c33d657c4d1)
![Imagen de WhatsApp 2024-05-26 a las 10 41 24_b35b6738](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/02eb5044-fd88-4907-a76b-428eead9b6b5)

Resultado de dockerizar la aplicación en una Imagen.

![Imagen de WhatsApp 2024-05-26 a las 10 43 41_efd2b189](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/0ff674ef-383b-43ec-8c4d-a9ee7e28270a)

Despliegue de la imagen back_fastapi-web en un contenedor.

![Imagen de WhatsApp 2024-05-26 a las 10 43 32_81aa9a55](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/9d34d8e7-5ad5-4e5a-ba8d-2021026c1473)


## Conclusiones 

El presente proyecto demuestra la funcionalidad del uso y comunicación de una aplicación móvil con el backend API que usa funciones PYCUDA, siendo usado Fast API para que los llamados POST HTTP puedan ser accedidos desde la aplicación móvil, esta aplicación siendo desarrollada con Flutter, una herramienta de desarrollo de aplicaciones multidispositivo, en este caso se usó para dispositivos Android, el cual a partir de este se generó la APK para una instalación de la aplicación más intuitiva sin el uso de software como Android Studio.

Un tema aparte fue la comunicación con el Backend donde se encontraba la comunicación a las funciones PyCuda de convolución de imágenes y el CRUD de login/register de los usuarios a la base de datos, esto por medio de librerías HTTP de los archivos .dart usados para la funcionalidad y diseño de la aplicación. Con esto es posible llamar funciones POST de la API por medio de las URL, aparte se agregó un sistema de configuración de red que ayudara a definir la IP de conexión a la API, esto enviará los datos que se obtengan de la aplicación( imágenes, usuario, contraseñas, etc.) para ser procesados en la API y regresados como un resultado favorable, con ello es posible enviar imágenes para aplicar los filtros a elección del usuario, seleccionando y aplicando la aplicación mostrará el resultado desplegado a visualización del usuario.

Finalmente, cuando se finalizó la aplicación de forma local, la dockerizacion fue lo siguiente para crear una IMAGEN de la API para posteriormente ser desplegada como contenedor, esto con la finalidad de que la aplicación sea usada desde cualquier y para cualquier forma en el que el backend este siendo desplegado. Con ello se ha finalizado el proceso de aprendizaje, aplicación y construcción de una aplicación móvil para la convolución de imágenes basados en algoritmos de PyCuda.


## Recomendaciones

Intentar probar nuevas formas de comunicación HTTP de forma dinámica, probablemente con servidores web o bluetooth.

Indagar más sobre cómo guardar algunos datos dentro de la base de datos, como detalles a fondo o reportes complejos.

Adentrarse más en codigo de PyCuda para aplicar operaciones más complejas, aprovechando el procesamiento de GPU de la API del Back.

Investigar más sobre los archivos DART y su diseño para trabajos futuros.

Probar otras aplicaciones flutter para la generación de aplicaciones.
