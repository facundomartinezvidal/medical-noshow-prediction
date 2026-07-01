## Why

El ausentismo a turnos médicos (no-show) ronda el 20–30 % de los turnos otorgados; en el dataset de trabajo es del **20,2 %** (22.319 de 110.527 turnos). Cada ausencia genera un doble costo: una franja del profesional queda ociosa y otro paciente que sí habría asistido no pudo ocupar ese lugar. Hoy la institución no tiene información anticipada sobre quién faltará, por lo que no puede accionar de forma preventiva.

Este cambio construye —desde cero, ya que el repo no tiene código— la implementación en Python que predice, al momento de agendar, la probabilidad de no-show de cada turno, para habilitar recordatorios dirigidos y sobreturnos controlados. Es el entregable del Trabajo Práctico Obligatorio de Ciencia de Datos (UADE, Grupo 7).

## What Changes

- Se agrega un **pipeline de datos** que ingiere el dataset de turnos (Kaggle, `no-show-dataset.csv`, 110.527 filas), valida su esquema, limpia y hace ingeniería de características (lead time, día/mes del turno, edad, comorbilidades, etc.).
- Se integra una **segunda fuente de datos** distinta: clima horario INMET 2016 (`weather-dataset.csv`), filtrando la estación **A612 (Vitória, ES)**, agregando horario→diario y cruzando por la fecha del turno (expectativa superadora).
- Se entrena un **modelo de clasificación binaria** (Árbol de Decisión y Random Forest) que devuelve la probabilidad de no-show por turno, evaluado con precision, recall, F1, ROC-AUC y PR-AUC (por el desbalance ~80/20).
- Se construye una **aplicación funcional interactiva en Streamlit** que traduce la predicción en valor: dado un turno, muestra la probabilidad de ausencia y la acción recomendada (recordatorio / sobreturno), con modo individual y modo lote (agenda del día).
- Se agregan artefactos de soporte: notebook de EDA + storytelling, diagrama de arquitectura, `requirements.txt` y `README`.

## Capabilities

### New Capabilities
- `data-pipeline`: ingesta y validación del dataset de turnos, limpieza e ingeniería de características, y armado del dataset procesado reutilizable por el modelo y la app.
- `weather-integration`: incorporación de la fuente secundaria de clima (INMET A612 Vitória), agregación horario→diario y cruce por fecha del turno.
- `noshow-modeling`: entrenamiento y evaluación de los modelos DT/RF, generación de la probabilidad de no-show y persistencia del artefacto entrenado con su pipeline de preprocesado.
- `interactive-app`: aplicación Streamlit que consume el modelo y entrega la probabilidad de no-show y la acción recomendada, en modo turno individual y modo lote.

### Modified Capabilities
<!-- Ninguna: no existen specs vivientes previas. Todos los requisitos son nuevos (ADDED). -->

## Impact

- **Código nuevo:** paquete `src/noshow/` (`config`, `data_load`, `weather`, `features`, `preprocess`, `train`, `evaluate`, `predict`), `app/streamlit_app.py`, `notebooks/01_eda_modeling.ipynb`, `docs/architecture.md`.
- **Datos:** `no-show-dataset.csv` (10,7 MB) y `weather-dataset.csv` (437 MB) ya en el repo → van a `.gitignore`; se versiona solo el diario de clima derivado y/o cómo regenerarlo.
- **Dependencias nuevas:** `pandas`, `scikit-learn`, `streamlit`, `matplotlib`/`seaborn`, `joblib`, `jupyter`.
- **Entorno:** Python 3.14 (verificar wheels; fallback venv 3.12).
- **Artefactos generados:** `models/*.joblib` + metadata de métricas, `reports/figures/`.
