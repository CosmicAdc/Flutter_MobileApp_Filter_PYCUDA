# ReactNative_MobileApp_Filter_PYCUDA
 Parallel Computing Integrator Project

 ## Carrera: Computacion

 ## Asignatura: Computacion Paralela

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
