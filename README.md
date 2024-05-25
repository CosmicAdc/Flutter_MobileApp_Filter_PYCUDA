# ReactNative_MobileApp_Filter_PYCUDA
 Parallel Computing Integrator Project

 ## Carrera: Computacion

 ## Asignatura: Computacion Paralela

 ## Integrantes:

 Andres Alba
  
 Christian Buestan

 Santiago Torres



 ## Objetivo alcanzado: 
Desarrollar una aplicacion flutten de edicion de imagenes por medios de filtros personalizados en base a algoritmos de convolucion utilizando PyCUDA, aprovechando la potencia de procesamiento GPU, esta convolucion se realiza en una API con comunicacion a una Base de datos; la aplicacion movil llama a esta API para mostrar los resultados, esta API es dockerizada y ejecutada.

## Actividades desarrolladas
1. Se programo 3 filtros personalizados. Siendo un filtro respectivo a icono de la UPS. Estos filtros fuero en base a algoritmos utilizando PyCUDA y el procesamiento de la GPU
   1 Filtro del logo de la UPS que se combina con la imagen de forma que quede como una marca de agua.
   1 Filtro circular de radio modificable, enfocando solo el contenido de la imagen dentro del radio.
   1 Filtro Fuxia que modifica los pixeles de la imagen para que predomine el canal Rojo, el canal verde se disminuya y que el canal azul se medialize.
3. Se desarrollo una API que use estos algoritmos en llamados REQUEST POST para aplicar uno de los 3 filtros a una imagen dada y que regrese el resultado de este proceso.
   En esta API se llaman archivos PYTHON para cada filtro a aplicar en la imagen, la imagen es cargada una funcion POST, con ello el usario llamara la funcion POST llamando a la imagen y seleccionando el filtro respectivo a aplicar, el resultado es la imagen resultante y el tiempo de ejecucion del proceso.
5. Se realizo una comunicacion a base de datos POSTGRESQL para hacer 2 procesos, login y register de usarios, aplicados dentro de la API.
   Este proceso fue mediante CRUD de un usuario y la coneccion al DB, donde se puede registrar un nuevo usuario, adjuntandoes a la base de datos, tambien con login el usuario iniciara sesion dentro de la aplicacion, si es exitoso el sistema devolvera el nombre de usuario y su ID respectivo, estos procesos de Register y login son accedidos por URL HTTP desdfe la aplicacion movil.
7. Se desarrollo una aplicacion movil usando FLutten y comunicaciones por medio de URL de los request de la API, esto para las paginas de login, register y convolucion de imagenes por medio de imagenes del dispositivo movil.
   Se usaron archivos DART para el diseño y funcionalidad de coneccion HTTP a las funciones de PYCUDA y sistema de Login/Register de la API, esto ademas de un diseño intuitivo para el usario, asi como el manejo de errores de coneccion a la API para la garantizar la disponibilidad de la aplicacion. Esto permitira generar la APK de la aplicacion para poder instalarla en los sistemas moviles. Las paginas de login y register son la pagina principal para verificar que el usuario este autorizado, luego con ello se guiara a la pagina de convolusiones de imagenes donde se capturara la imagen con la camara del dispositovo, posteriormente se seleccionara el tipo de filtro a aplicara aplicar para ser guardado y mostrado en la misma aplicacion.
9. Dockerizacion de la API y la base de datos.
10. Generacion de la APK.

## Resultados obtenidos
Pagina principal LOGIN de un nuevo usuario:

![Imagen de WhatsApp 2024-05-24 a las 12 18 19_050fb970](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/358f772a-0763-4332-9fa8-b1a9d69be423)

Pagina de registro de usuarios dentro de la base de datos.

![Imagen de WhatsApp 2024-05-24 a las 12 18 19_74e8bf17](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/a3457e1e-8e60-45ed-b2ca-27a25081312b)

Manejo de errores con respecto a la comunicacion con el BackEnd API.

![Imagen de WhatsApp 2024-05-23 a las 04 35 09_1efcbdef](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/3b477b5a-4268-453f-9ef0-4f6fbadd1908)

Generacion del DockerFile y Dockerizacion de la API.

![Imagen de WhatsApp 2024-05-24 a las 00 30 21_fcc247fe](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/726118c7-b3d9-4ef9-b0e9-648d4a1bae56)

Visualizacion de la Pagina de presentacion de las Imagenes Convolucionadas y el Boton de realizar una nueva convolucion.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m10s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/71ee71a0-c11f-45bf-a28e-71ec2bf1e95f)

Uso de la camara con Flutter para obtener la imagen a procesar.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m13s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/2ab5689f-c802-4af3-a4a6-b3bf8f303efa)

Seleccion de filtro circular a aplicar sobre la imagen tomada.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m19s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/59b62d49-cdf7-4878-bf14-9cdc9d7be9be)

Seleccion de filtro marea a aplicar en la imagen tomada.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m21s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/f0d6b434-b111-4c9c-b427-6fa15f4b2de0)

Seleccion de filtro UPS a aplicar en la imagen tomada.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m22s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/66d2ffa0-cbe7-4d53-acd1-8434cec77f0b)

-Cuando se haya definido el filtro a aplicar se selecciona el boton de enviar del menu para guardar el resultado con un nombre de archivo ingresado por el usuario.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m30s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/e2fa85c2-640d-4031-a0ea-81ca4c44eaba)

Guardado exitoso de la imagen y su visualizacion en la aplicacion movil.

![Video de WhatsApp 2024-05-23 a las 00 16 11_e28eeb8a - frame at 0m46s](https://github.com/CosmicAdc/Flutter_MobileApp_Filter_PYCUDA/assets/84852007/b166a75a-8489-4916-81a2-bd11032e2e76)

Resultado de dockerizar la Aplicacion

## Conclusiones 

El presente proyecto demuestra la funcionalidad del uso y comunicacion de una aplicacion movil con el backend API que usa funciones PYCUDA, siendo usado Fast API para que los llamados POST HTTP puedan ser accedidos desde la aplicacion movil, esta aplicacion siendo desarrollada con Flutten, una herramienta de desarrollo de aplicaciones multidispositivo, en este caso se uso para dispositvios android, el cual a partir de este se genero la APK para una instalacion de la aplicacion mas intuitiva sin el uso de software como Android Studio.

Un tema aparte fue la comunicacion con el Backend donde se encontraba la comunicacion a las funciones PyCuda de convolucion de imagenes y el CRUD de loging/register de los usuarios a la base de datos, esto por medio de librerias HTTP de los archivos .dart usados para la funcionalidad y diseño de la aplicacion. Con esto es posible llamar funciones POST de la API por medio de las URL, aparte se agrego un sistema de configuracion de red que ayudara a definir la IP de coneccion a la API, esto enviara los datos que se obtengan de la aplicacion(Imagen,usuario,contraseñas, etc.) para ser procesados en la API y regresados como un resultado favorable, con ello es posible enviar imagenes para aplicar los filtros a eleccion del usuario, seleccionando y aplicando la aplicacion mostrara el resultado desplegado a visualizacion del usuario.

Finalmente cuando se finalizo la aplicacion de forma local, la dockerizacion fue lo siguiente para crear una IMAGEN de la API para posteriormente ser deplegada como contenedor, esto con la finalidad de que la aplicacion sea usada desde cualquier y para cualquier forma en el que el backend este siendo desplegado. Con ello se ha finalziado el proceso de aprendizaje, aplicacion y construccion de una aplicacion movil para la convolucion de imagenes basados en algoritmos de PyCuda. 

## Recomendaciones

Intentar probar nuevas formas de comunicacion HTTP de forma dinamica, probablemente con servidores web o bluetooth.

Indagar mas sobre como guardar algunos datos dentro de la base de datos, como detalles a fondo o reportes complejos.

Adentrarse mas en codigo de PyCuda para aplicar operaciones mas complejas, aprovechando el procesamiento de GPU de la API del Back.

Investigar mas sobre los archivos DART y su diseño para trabajos futuros.

Probar otras aplicaciones Flutten para la generacion de aplicaciones.
