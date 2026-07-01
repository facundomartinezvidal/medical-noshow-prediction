## ADDED Requirements

### Requirement: Preprocesado sin fuga de datos
El sistema SHALL encapsular el preprocesado (imputación, codificación de categóricas, escalado) en un pipeline de scikit-learn que se ajusta únicamente sobre los datos de entrenamiento y se persiste junto al modelo.

#### Scenario: Ajuste solo en train
- **WHEN** se entrena el modelo
- **THEN** el `ColumnTransformer`/pipeline SHALL calcular sus parámetros (categorías, medias de imputación, etc.) usando exclusivamente el conjunto de entrenamiento, nunca el de test

#### Scenario: Reutilización en inferencia
- **WHEN** la aplicación scorea un turno nuevo
- **THEN** SHALL usar el mismo pipeline persistido, garantizando idéntica transformación que en entrenamiento

#### Scenario: Codificación de categóricas
- **WHEN** se codifican variables categóricas nominales (género, barrio, grupo etario)
- **THEN** el sistema SHALL usar One-Hot Encoding (por ejemplo `OneHotEncoder(handle_unknown="ignore")` o `get_dummies(drop_first=True)`), evitando Label Encoding en nominales para no inyectar un orden artificial, conforme a la cátedra; los modelos de árbol NO requieren escalado

### Requirement: Entrenamiento de modelos de clasificación
El sistema SHALL entrenar un Árbol de Decisión y un Random Forest sobre el dataset procesado, usando una partición train/test estratificada por la clase objetivo y manejando el desbalance de clases (~80/20).

#### Scenario: Partición estratificada
- **WHEN** se divide el dataset en train y test (hold-out)
- **THEN** la proporción de la clase `no_show` SHALL preservarse en ambos conjuntos (split estratificado)

#### Scenario: Manejo del desbalance
- **WHEN** se aborda el desbalance de clases (~80/20)
- **THEN** el sistema SHALL basarse en muestreo estratificado (`stratify=y` en el split y `StratifiedKFold` en la validación) y en métricas apropiadas (F1 y recall de la clase minoritaria), conforme a lo enseñado en la cátedra; `class_weight="balanced"` es una palanca OPCIONAL, no el mecanismo principal

### Requirement: Validación cruzada estratificada
El sistema SHALL usar validación cruzada `StratifiedKFold` (con `shuffle=True` y `random_state` fijo) sobre el conjunto de entrenamiento para comparar los modelos y estimar el desempeño de forma robusta, preservando la proporción de clases en cada fold.

#### Scenario: Proporción de clases por fold
- **WHEN** se generan los k folds de validación cruzada
- **THEN** cada fold SHALL conservar (aproximadamente) el ~80/20 de la clase `no_show` del conjunto completo

#### Scenario: Métricas agregadas entre folds
- **WHEN** finaliza la validación cruzada de un modelo
- **THEN** el sistema SHALL reportar la media y el desvío estándar de las métricas (por ejemplo F1 y ROC-AUC) a través de los folds, no un único valor de un solo corte

#### Scenario: Selección de modelo con CV
- **WHEN** se comparan Árbol de Decisión y Random Forest
- **THEN** la elección del modelo final SHALL basarse en el desempeño de la validación cruzada sobre train, reservando el hold-out de test para la evaluación final

### Requirement: Probabilidad de no-show por turno
El sistema SHALL producir, para cada turno, una probabilidad de ausencia en el rango [0, 1] mediante `predict_proba`, no solo una etiqueta.

#### Scenario: Salida probabilística
- **WHEN** se scorea un turno
- **THEN** el sistema SHALL retornar un número entre 0 y 1 que representa la probabilidad estimada de no-show

### Requirement: Evaluación con métricas adecuadas al desbalance
El sistema SHALL evaluar los modelos con `classification_report` (accuracy, precision, recall y F1 por clase), la matriz de confusión visualizada con `ConfusionMatrixDisplay`, y ROC-AUC (`roc_auc_score`), sin usar accuracy como métrica principal por el desbalance. PR-AUC es una métrica OPCIONAL adicional (no cubierta por el material de cátedra).

#### Scenario: Reporte de métricas
- **WHEN** finaliza el entrenamiento
- **THEN** el sistema SHALL producir el `classification_report`, la matriz de confusión (`ConfusionMatrixDisplay`) y el ROC-AUC sobre el conjunto de test, con una comparativa entre Árbol de Decisión y Random Forest

#### Scenario: Razonamiento sobre el tipo de error
- **WHEN** se interpreta la matriz de confusión en el contexto del negocio
- **THEN** el análisis SHALL discutir explícitamente el costo relativo de un falso negativo (no-show no detectado) frente a un falso positivo, para justificar la elección del umbral y la métrica priorizada

#### Scenario: Persistencia del artefacto
- **WHEN** se selecciona el modelo final
- **THEN** el sistema SHALL guardar el modelo entrenado (pipeline + clasificador) en `models/` en formato joblib, junto con la metadata de métricas

### Requirement: Control de sobreajuste e interpretabilidad
El sistema SHALL controlar el sobreajuste del Árbol de Decisión mediante poda (hiperparámetros como `max_depth` y `min_samples_leaf`) y SHALL exponer la interpretabilidad de los modelos (visualización del árbol e importancia de variables del Random Forest).

#### Scenario: Poda del árbol
- **WHEN** se entrena el Árbol de Decisión
- **THEN** el sistema SHALL limitar su complejidad (por ejemplo `max_depth`/`min_samples_leaf`) y evidenciar la diferencia de desempeño entre entrenamiento y test para mostrar el control de overfitting; el Random Forest NO se poda

#### Scenario: Importancia de variables
- **WHEN** se entrena el Random Forest
- **THEN** el sistema SHALL reportar `feature_importances_` en un gráfico de barras ordenado, para el storytelling técnico
