# Predicción de no-show en turnos médicos

Trabajo Práctico Obligatorio de Ciencia de Datos — **UADE, Grupo 7**.

Implementación en Python de un pipeline de datos completo (turnos médicos de Vitória,
ES + clima INMET), un modelo de clasificación (Árbol de Decisión / Random Forest) que
predice la probabilidad de **no-show** (ausentismo a turnos médicos ya agendados), y una
aplicación interactiva en Streamlit que traduce esa probabilidad en una acción de
negocio concreta.

## Propuesta de valor

El ausentismo a turnos ronda el **20,2 %** en el dataset de trabajo (22.319 de 110.527
turnos). Cada no-show implica un doble costo: una franja del profesional queda ociosa y
otro paciente en lista de espera pierde la oportunidad de ocuparla. Este proyecto estima,
**al momento de agendar el turno**, la probabilidad de no-show para habilitar dos
acciones preventivas:

1. **Recordatorio reforzado** en turnos de riesgo medio.
2. **Sobreturno controlado** en turnos de riesgo alto.

de forma que los recursos limitados de la institución (llamados, sobreturnos) se
prioricen donde más impacto tienen, en vez de aplicarse de forma uniforme.

## Estructura del repositorio

```
.
├── app/
│   └── streamlit_app.py         # App interactiva (modo individual y modo lote)
├── data/
│   ├── raw/                     # CSV crudos (no versionados, ver "Datos" abajo)
│   ├── external/                # Clima diario A612 ya agregado (sí versionado, chico)
│   └── processed/                # Dataset procesado cacheado (generado, no versionado)
├── docs/
│   └── architecture.md          # Diagrama de arquitectura (Mermaid) + descripción del stack
├── models/
│   ├── model.joblib              # Pipeline (preprocesado + clasificador) entrenado
│   └── metrics.json              # Métricas de CV y hold-out del modelo persistido
├── notebooks/
│   └── 01_eda_modeling.ipynb     # EDA + modelado + storytelling, estructurado en CRISP-DM
├── openspec/                     # Propuesta, diseño y specs del cambio (spec-driven dev)
├── reports/
│   └── figures/                  # Figuras exportadas por el notebook (para la presentación)
├── src/
│   └── noshow/                   # Paquete Python: config, data_load, weather, features,
│                                  # preprocess, train, evaluate, predict
├── tests/                        # Suite de pytest (capa de datos y de modelado)
├── requirements.txt
└── README.md
```

## Setup

### 1. Entorno virtual

Se recomienda **Python 3.13**. Al momento de escribir este README, Python 3.14 es
demasiado reciente y algunas dependencias científicas (`pandas`, `scikit-learn`,
`streamlit`) pueden no tener wheels precompilados todavía, lo que obliga a compilar
desde fuente o directamente falla la instalación. Si `python3.14 -m venv .venv` da
problemas al instalar `requirements.txt`, usar 3.13 (o, como segundo fallback, 3.12).

```bash
python3.13 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .               # instala el paquete `noshow` (editable) para que
                               # `python -m noshow.*` y la app lo encuentren desde la raíz
```

### 2. Datos

El repositorio **no versiona** los datasets crudos por su tamaño (10,7 MB y ~437 MB
respectivamente). Hay que colocarlos manualmente en `data/raw/`:

```
data/raw/no-show-dataset.csv     # Kaggle: Medical Appointment No Shows (110.527 turnos)
data/raw/weather-dataset.csv     # INMET: datos horarios 2016, todas las estaciones (~437 MB)
```

El clima horario se procesa **por chunks** (nunca se carga completo en memoria) y se
filtra a la estación **A612 (Vitória, ES)** — la ciudad de origen de los turnos. El
resultado ya agregado a diario (`data/external/weather_daily_a612.csv`, ~22 KB) **sí
está versionado** en el repo, así que no hace falta tener `weather-dataset.csv` para
correr el pipeline si ese caché ya existe (se regenera automáticamente si se borra,
siempre que el crudo esté presente en `data/raw/`).

## Uso

Todos los comandos asumen el venv activado y se ejecutan desde la raíz del repo.

### Construir el dataset procesado

```bash
python -m noshow.features
```

Carga los turnos, cruza el clima y aplica la ingeniería de características; cachea el
resultado en `data/processed/appointments_processed.csv`.

### Entrenar el modelo

```bash
python -m noshow.train
```

Entrena Árbol de Decisión (podado) y Random Forest (sin podar), los compara por
validación cruzada estratificada, elige el modelo final por ROC-AUC y persiste:

- `models/model.joblib` — pipeline completo (preprocesado + clasificador).
- `models/metrics.json` — métricas de CV y del hold-out de test.

### Notebook de EDA + modelado

```bash
jupyter lab notebooks/01_eda_modeling.ipynb
```

O para ejecutarlo de punta a punta sin abrir la UI (validación headless):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_modeling.ipynb
```

El notebook sigue el framework **CRISP-DM** (comprensión del negocio → comprensión y
preparación de los datos → EDA/storytelling → modelado → evaluación → conclusión) y
exporta las figuras clave a `reports/figures/`.

### App interactiva

```bash
streamlit run app/streamlit_app.py
```

Requiere que `models/model.joblib` exista (ver "Entrenar el modelo" arriba). Ofrece un
modo de turno individual (inputs → probabilidad de no-show + acción recomendada) y un
modo lote (subir un CSV de agenda → tabla rankeada por riesgo).

### Tests

```bash
pytest
```

La suite usa un dataset sintético pequeño para los tests de modelado (nunca lee el
dataset real de 110.521 filas dentro de un test unitario), por lo que corre rápido y no
requiere tener los CSV crudos descargados.

## Limitaciones

- **Datos de 2016, una única ciudad (Vitória, ES, Brasil) y pre-COVID**: los patrones de
  asistencia a salud pueden haber cambiado estructuralmente desde entonces (telesalud,
  nuevos hábitos post-pandemia). El modelo no debe aplicarse tal cual a otra geografía o
  período sin revalidar.
- **Concept drift**: los factores que explican el no-show cambian con el tiempo
  (estacionalidad, políticas de turnos, composición socioeconómica de los pacientes). Se
  recomienda reentrenar periódicamente y monitorear ROC-AUC/recall en producción.
- **Ventana de clima acotada** (~1,5 meses de 2016, la cobertura real de los turnos), lo
  que limita la variabilidad climática observada.
- El efecto de `SMS_received` sobre el no-show está **confundido con el lead time** (se
  envía más a turnos agendados con más anticipación): no se puede interpretar
  causalmente sin un diseño experimental controlado. Ver el caveat explícito en el
  notebook (Sección 4).
- Es un modelo **académico** (TPO), no un sistema desplegado en producción: no hay
  monitoreo continuo, reentrenamiento automático ni validación con feedback real de los
  recordatorios/sobreturnos aplicados.

## Metodología

El proyecto sigue **CRISP-DM** como framework de ciencia de datos (ver el notebook,
estructurado explícitamente en sus fases) combinado con **gestión ágil** y
**versionado en GitHub** (branch de feature, commits convencionales por módulo). El
diseño técnico completo — incluyendo las decisiones de modelado y su alineación con el
material de cátedra (Árbol de Decisión podado + Random Forest sin podar,
`StratifiedKFold`, One-Hot Encoding, evaluación con `classification_report` +
`ConfusionMatrixDisplay` + ROC-AUC, razonamiento explícito de FN vs FP) — está
documentado en `openspec/changes/noshow-prediction/` (`proposal.md`, `design.md`,
`tasks.md`), siguiendo un enfoque de **spec-driven development**.
