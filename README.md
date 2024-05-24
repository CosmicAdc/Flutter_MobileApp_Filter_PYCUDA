# ReactNative_MobileApp_Filter_PYCUDA
 Parallel Computing Integrator Project

 ## Carrera: Computacion

 ## Asignatura: Computacion Paralela

 ## Objetivo alcanzado: 
Desarrollar una aplicacion flutten de edicion de imagenes por medios de filtros personalizados en base a algoritmos de convolucion utilizando PyCUDA, aprovechando la potencia de procesamiento GPU, esta convolucion se realiza en una API con comunicacion a una Base de datos; la aplicacion movil llama a esta API para mostrar los resultados, esta API es dockerizada y ejecutada.

## Actividades desarrolladas
1. Se programo 3 filtros personalizados. Siendo un filtro respectivo a icono de la UPS. Estos filtros fuero en base a algoritmos utilizando PyCUDA y el procesamiento de la GPU
2. Se desarrollo una API que use estos algoritmos en llamados REQUEST POST para aplicar uno de los 3 filtros a una imagen dada y que regrese el resultado de este proceso
3. Se realizo una comunicacion a base de datos POSTGRESQL para hacer 2 procesos, login y register de usarios, aplicados dentro de la API.
4. Se desarrollo una aplicacion movil usando FLutten y comunicaciones por medio de URL de los request de la API, esto para las paginas de login, register y convolucion de imagenes por medio de imagenes del dispositivo movil.
5. Dockerizacion de la API y la base de datos.
6. Generacion de la APK.

## Resultados obtenidos
