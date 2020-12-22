# Ejercicio descriptores

El objetivo de este ejercicio es trabajar con la descripción de objetos y texturas usando HoG y GLCM en imágenes.


 
## Enunciado

Se debe resolver un problema de clasificación de imágenes, desde dos puntos de vista diferente.

Seguiremos los siguientes pasos:

- Utilizar un conjunto de datos en el que podamos encontrar dos o más clases. Un ejemplo válido (aunque poco emocionante) sería el ya conocido MNIST de números,  
otra posibilidad es usar una versión actualizada, [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist) de Zalando (más recomendable) y usar alrededor de 4-6 clases.

- **Generar** un conjunto de datos donde las clases estén representadas de igual manera (podemos pensar en un conjunto de 100 muestras por clase).

- **Describir** cada imágen usando:
    - GLCM como un histograma.
    - Un vector con almenos 4 medidas que se pueden extraer de las GLCM.
    - HOG.
    
- Dedicar un 80% de los datos al conjunto de entrenamiento y un 20% al conjunto de test.

- Usar un KNN ([*K-Nearest-Neighbor*](https://scikit-learn.org/stable/modules/neighbors.html#classification)) como clasificador. Realizar un **aprendizaje** para cada descriptor anteriormente mencionado. Para aquellos descriptores que són histogramas, la medida de distancia evidentemente debe estar en histograma. [tutorial](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

- **Predecir** las clases del conjunto de test. Como resultado mostrar la matriz de confusión y la *accuracy* obtenida.
- **Discutir** los resultados obtenidos.


## Entrega


Este ejercicio se entrega en formato *notebook(s)*, preferiblemente en formato pdf. En el, se debe poder seguir todo el proceso y finalmente ejemplos donde se discutan casos de clasificación correcta e incorrecta

La entrega se realizará el nodo correspondiente del aula digital de la asignatura.


Este es un ejercicio evaluable.

## Recursos

**Scikit** Posee herramientas para:

Separar un conjunto de datos en entrenamiento y test: [enlace](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

KNN: [enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

Matriz de confusión [enlace](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix).

Métricas [enlace](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).

**OpenCV** provee funciones para el cálculo de HoG como de sus métricas,
[enlace](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html).
 

**Scikit-image** provee funciones tanto para el cálculo de las GLCM como de sus métricas,
[enlace](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html).
 