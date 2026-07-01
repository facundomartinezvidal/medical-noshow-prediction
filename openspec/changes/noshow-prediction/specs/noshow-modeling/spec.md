## ADDED Requirements

### Requirement: Preprocesado sin fuga de datos
El sistema SHALL encapsular el preprocesado (imputación, codificación de categóricas, escalado) en un pipeline de scikit-learn que se ajusta únicamente sobre los datos de entrenamiento y se persiste junto al modelo.

#### Scenario: Ajuste solo en train
- **WHEN** se entrena el modelo
- **THEN** el `ColumnTransformer`/pipeline SHALL calcular sus parámetros (categorías, medias de imputación, etc.) usando exclusivamente el conjunto de entrenamiento, nunca el de test

#### Scenario: Reutilización en inferencia
- **WHEN** la aplicación scorea un turno nuevo
- **THEN** SHALL usar el mismo pipeline persistido, garantizando idéntica transformación que en entrenamiento

### Requirement: Entrenamiento de modelos de clasificación
El sistema SHALL entrenar un Árbol de Decisión y un Random Forest sobre el dataset procesado, usando una partición train/test estratificada por la clase objetivo y manejando el desbalance de clases (~80/20).

#### Scenario: Partición estratificada
- **WHEN** se divide el dataset en train y test
- **THEN** la proporción de la clase `no_show` SHALL preservarse en ambos conjuntos (split estratificado)

#### Scenario: Manejo del desbalance
- **WHEN** se instancia cada clasificador
- **THEN** el sistema SHALL aplicar una estrategia para el desbalance (por ejemplo `class_weight="balanced"`)

### Requirement: Probabilidad de no-show por turno
El sistema SHALL producir, para cada turno, una probabilidad de ausencia en el rango [0, 1] mediante `predict_proba`, no solo una etiqueta.

#### Scenario: Salida probabilística
- **WHEN** se scorea un turno
- **THEN** el sistema SHALL retornar un número entre 0 y 1 que representa la probabilidad estimada de no-show

### Requirement: Evaluación con métricas adecuadas al desbalance
El sistema SHALL evaluar los modelos con precision, recall, F1, ROC-AUC y PR-AUC, y reportar la matriz de confusión, evitando usar accuracy como métrica principal.

#### Scenario: Reporte de métricas
- **WHEN** finaliza el entrenamiento
- **THEN** el sistema SHALL imprimir/persistir precision, recall, F1, ROC-AUC y PR-AUC sobre el conjunto de test, y una comparativa entre Árbol de Decisión y Random Forest

#### Scenario: Persistencia del artefacto
- **WHEN** se selecciona el modelo final
- **THEN** el sistema SHALL guardar el modelo entrenado (pipeline + clasificador) en `models/` en formato joblib, junto con la metadata de métricas
