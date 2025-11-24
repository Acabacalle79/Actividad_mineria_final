*INTEGRANTES*

Jesus Ogliastri Reyes,
Juan Peña Mendoza,
Emanuel Marriaga Ramos, 
Pedro Calderon Heras




Link de video explicativo del código:https://youtu.be/z3ZtOkmuLyU

Metodología del Proyecto — Iris Species Classification

La metodología que aplicamos sigue el ciclo completo de un proyecto de minería de datos, desde la comprensión inicial del dataset hasta la implementación del modelo y su visualización en un dashboard interactivo.

____Comprensión de los datos____
El proyecto utiliza el dataset Iris, uno de los referentes más ampliamente usados en Machine Learning, compuesto por:

-150 registros

-3 especies: Setosa, Versicolor y Virginica

-4 características numéricas:

-Sepal Length

-Sepal Width

-Petal Length

-Petal Width

En esta fase se realizaron:

-Estadísticas descriptivas.

-Revisión de distribución de variables mediante histogramas.

-Gráficos de dispersión para identificar relaciones entre pares de variables.

-Análisis preliminar que muestra que las especies Setosa y Virginica se separan claramente, mientras que Versicolor se solapa parcialmente con Virginica.

-Objetivo de esta etapa: comprender el comportamiento del dataset y detectar qué variables son más relevantes para la clasificación.


____Preprocesamiento de datos____

El dataset Iris no contiene valores faltantes, por lo que las principales acciones fueron:

Dividir los datos usando train_test_split con un 80% para entrenamiento y 20% para prueba.

Mantener un random_state=42 para garantizar reproducibilidad.

No fue necesario aplicar escalado ya que el modelo elegido (Random Forest) no lo requiere.

Resultado: datos limpios y listos para el modelado.


Entrenamiento del modelo
Se seleccionó el algoritmo Random Forest Classifier debido a que:

-Maneja muy bien problemas de clasificación multiclase.

-Reduce el riesgo de sobreajuste comparado con un solo árbol.

-Funciona correctamente sin necesidad de escalar los datos.

-Es interpretable y rápido de entrenar.

Pasos realizados:

1 Entrenar el modelo con los datos de entrenamiento.

2 Ajustar hiperparámetros básicos como n_estimators y max_depth (opcional).

3 Guardar el modelo final usando joblib para integrarlo en el dashboard.


____Evaluación del modelo____
Se evaluó el desempeño en el conjunto de prueba usando las métricas:

-Accuracy

-Precision

-Recall

-F1-score

Matriz de confusión

Estas métricas permiten identificar:

-Qué tan bien generaliza el modelo.

-En qué especies comete más errores.

-Si existe sesgo hacia alguna clase.

-Resultado esperado: valores altos en todas las métricas y una matriz con baja confusión entre Versicolor y Virginica.


____Desarrollo del dashboard en Streamlit____
El dashboard fue diseñado para ser interactivo, intuitivo y completo.
Incluye:

a) Panel de métricas del modelo

Muestra accuracy, precision, recall y F1 de manera clara.

b) Visualizaciones

Histogramas por característica

Scatter plots

Gráfico 3D interactivo que permite visualizar la posición de una nueva muestra

c) Panel de predicción

El usuario puede ingresar:

-Sepal length

-Sepal width

-Petal length

-Petal width

Y el sistema:

Predice la especie.

Muestra el punto directamente en el gráfico 3D junto al dataset.

Objetivo: comunicar resultados de forma visual y permitir interacción con el modelo.
