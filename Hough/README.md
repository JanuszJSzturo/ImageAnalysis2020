# Ejercicio transformada Hough

El objetivo de este ejercicio es encontrar objetos circulares en una imagen utilizando la técnica de la transformada de Hough.

Se recomienda el uso de OpenCV para llevar a cabo esta tarea y  la lectura de este [tutorial](https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/)
 
 
## Enunciado
Se debe realizar un *script* que dadas la imágenes que podéis encontrar en la carpeta *input*, una imagen de entrada (en escala de grises) y genere el centro y el radio de todos los objetos circulares encontrados. 

Se desea generar solo una hipótesis por objeto, por lo tanto, los círculos que son concéntricos cercanos (es decir, tienen el mismo centro dentro de lo que considere que es una tolerancia razonable y no son radicalmente diferentes en tamaño) deben agruparse, ya que es probable que surjan del mismo objeto. 

El programa también debe mostrar la imagen de salida y dibujar los círculos ubicados en ella.

La primera imágen es relativamente sencilla de procesar, pero las dos siguiente presentan una dificultad más elevada. Es suficiente con conseguir el objetivo con la primera de ellas, las otras do se añaden como un reto al estudiante.

 
## Entrega

Este ejercicio se entrega en el nodo correspondiente del aula digital de la asignatura.

No es un ejercicio evaluable y lo corregiremos en clase.