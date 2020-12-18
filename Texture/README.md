# Ejercicio textura

El objetivo de este ejercicio es trabajar con la descripción de texturas usando GLCM en imágenes.

Se recomienda el uso de Scikit-image que provee de funciones tanto para el cálculo de las GLCM como de sus métricas,
[enlace](https://scikit-image.org/docs/dev/auto_examples/features_detection
/plot_glcm.html).
 
 
## Enunciado

Se debe realizar un *script* con las soguientes características:

- Implementar un KNN ([*K-Nearest-Neighbor*](https://scikit-learn.org/stable/modules/neighbors.html#classification)) 
para saber si dos imágenes son la *misma* (similares). 
- Utilizar GLCM como descriptor de cada imágen.
- Seleccionar como mínimo 3 medidas. 
- Utilizar un conjunto de datos en el que podamos encontrar dos o más clases.
Un ejemplo válido (aunque poco emocionante) sería el ya conocido MNIST de números, 
otra posibilidad es usar una versión actualizada [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist)
de Zalando. Otra idea sería usar imágenes de satélite ([aquí](https://github
.com/chrieke/awesome-satellite-imagery-datasets) tenéis algunos ejemplos.

- Debéis comparar utilizando alguna distancia basada en histogramas.


## Entrega


Este ejercicio se entrega en formato *notebook*, preferiblemente en formato pdf. En el se debe poder seguir todo el proceso y finalmente ejemplos donde se discutan casos de clasificación correcta e incorrecta

La entrega se realizará el nodo correspondiente del aula digital de la
asignatura (17 Enero).


Este es un ejercicio evaluable.