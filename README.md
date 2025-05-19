# Clasificación de Imágenes de Infecciones por Hongos con TensorFlow

## 📝 Descripción

Proyecto académico para la materia Desarrollo de aplicaciones avanzadas de ciencias computacionales. Este repositorio implementa un modelo de aprendizaje supervisado con TensorFlow para la clasificación de imágenes médicas en cinco clases distintas de infecciones por hongos.
Se ha desarrollado con la técnica de **Aprendizaje Supervisado**. Esta es una subárea del Machine Learning en la que el modelo se entrena a identificar patrones alimentándose de un conjunto de datos con etiquetas (en este caso, utilizando 5 categorías). Se busca que el sistema realice una tarea de clasificación donde pueda identificar a cuál de las 5 categorías pertenece cada una de las imágenes

## 📂 Sobre el Dataset

Recuperado del link: https://archive.ics.uci.edu/dataset/773/defungi
Autores: María Alejandra Vanegas Álvarez, Leticia Sopó, C. Sopo, F. Hajati, S. Gheisari

La estructura del dataset es la siguiente:
´´´
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
´´´

# 🧹 Preprocesamiento de Datos

Para preparar las imágenes antes de entrenar el modelo, se utilizó la clase ImageDataGenerator de TensorFlow con un reescalado de entre 0 y 1, dividiendo cada valor por 255. Esto con el propósito de permitir una mejor convergencia del modelo durante el entrenamiento. También, se hizo un redimensionamiento de las imágenes como el hecho en clase a un tamaño uniforme de 150x150 píxeles para asegurar la compatibilidad con la arquitectura de la red neuronal. Anteriormente, el modelo contaba con un recorte de las imágenes para hacer enfoque sólo de la zona de interés para el entrenamiento, algo que definitivamente apoyó bastante. Finalmente, se colocan los datos en batches y asigna etiquetas de clase en modo categórico debido a que esto es una clasificación multiclase (tenemos 5 clases) en vez de las usadas en clase que eran binarias (sólo 2 clasificaciones).

# 🔁 Data Augmentation

Para mejorar la generalización del modelo y evitar el sobreajuste debido al bias, la técnica de Data Augmentation. El proceso incluyó rotaciones de hasta 10 grados, desplazamientos horizontales de la imagen de hasta un 20%, zoom de hasta un 30% y volteo horizontal. Todo para permitir al modelo generar variantes de las imágenes alimentadas al inicio, enriqueciendo el dataset de entrenamiento sin necesidad de recolectar más datos. Al final, las imágenes aumentadas se guardaron en la carpeta 'augmented'.
