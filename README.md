# Clasificación de Imágenes de Infecciones por Hongos con TensorFlow

## Descripción

Proyecto académico para la materia Desarrollo de aplicaciones avanzadas de ciencias computacionales. Este repositorio implementa un modelo de aprendizaje supervisado con TensorFlow para la clasificación de imágenes médicas en cinco clases distintas de infecciones por hongos.
Se ha desarrollado con la técnica de **Aprendizaje Supervisado**. Esta es una subárea del Machine Learning en la que el modelo se entrena a identificar patrones alimentándose de un conjunto de datos con etiquetas (en este caso, utilizando 5 categorías). Se busca que el sistema realice una tarea de clasificación donde pueda identificar a cuál de las 5 categorías pertenece cada una de las imágenes

## Sobre el Dataset

La estructura del dataset es la siguiente:
```
DFungi Dataset
├── test
│ ├── H1 (3563 imágenes)
│ ├── H2 (1887 imágenes)
│ ├── H3 (667 imágenes)
│ ├── H5 (666 imágenes)
│ └── H6 (602 imágenes)
└── train
│ ├── H1 (891 imágenes)
│ ├── H2 (474 imágenes)
│ ├── H3 (162 imágenes)
│ ├── H5 (162 imágenes)
│ └── H6 (148 imágenes)
└── augmented (312 imágenes)
```
