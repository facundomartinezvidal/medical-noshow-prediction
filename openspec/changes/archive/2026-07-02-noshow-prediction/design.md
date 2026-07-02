## Context

TPO de Ciencia de Datos (UADE, Grupo 7). Dominio: no-show de turnos médicos. El repo parte sin código; hay que construir el proyecto de datos completo en Python. Dos fuentes reales ya presentes en el repo: turnos (`no-show-dataset.csv`, 110.527 filas, 20,2 % no-show) y clima horario INMET 2016 (`weather-dataset.csv`, 4,37M filas, 529 estaciones). Audiencia del trabajo: Gerencia Comercial y Técnica. Metodología marco: CRISP-DM + gestión ágil + versionado en GitHub.

Restricciones: Python 3.14 (entorno muy nuevo), archivo de clima de 437 MB, clases desbalanceadas, y la necesidad de una app funcional interactiva que materialice el valor.

## Goals / Non-Goals

**Goals:**
- Pipeline reproducible: crudo → validado → features → dataset procesado.
- Integrar clima como segunda fuente (superadora) cruzada por fecha del turno.
- Modelo DT/RF que emita probabilidad de no-show, evaluado con métricas robustas al desbalance.
- App Streamlit que traduzca la probabilidad en acción de negocio (recordatorio / sobreturno), modo individual y lote.
- Sin fuga de datos: preprocesado ajustado solo en train y persistido con el modelo.

**Non-Goals:**
- No es un sistema productivo desplegado ni un servicio en tiempo real.
- No se incorporan datos posteriores a 2016 (los turnos son de 2016; el clima debe ser del mismo período).
- No se optimiza exhaustivamente hiperparámetros más allá de lo necesario para el TP.
- No se modela historial causal complejo; el historial de no-show previo del paciente es opcional/avanzado y solo con cuidado de fuga temporal.

## Decisions

- **Streamlit para la app.** Estándar de facto para demos de data science en Python; permite inputs interactivos y visualización con poco código. Alternativas descartadas: Gradio (menos flexible para dashboard/storytelling), Flask+HTML (demasiado código para un TP).
- **Fuente de clima = INMET estación A612 (Vitória, ES).** Se validó que A612 es costera (presión ~1016 mB, temp media 24,8 °C), coherente con Vitória. Se agrega horario→diario y se cruza por `AppointmentDay`. Alternativa descartada: Open-Meteo API (innecesaria, ya hay datos reales locales).
- **Solo el clima del día del turno importa** (¿llovió → faltó?), no el del agendamiento → basta la ventana 2016-04-29 a 2016-06-08.
- **DT + RF con `class_weight="balanced"`** y split estratificado. Métrica principal: recall de la clase "falta" + F1/AUC; accuracy se evita por engañosa ante 80/20.
- **Pipeline sklearn (`ColumnTransformer`) persistido con el modelo** (joblib) → misma transformación en train y en la app, sin fuga.
- **Paquete `src/noshow/` modular** (`config, data_load, weather, features, preprocess, train, evaluate, predict`) + `app/` + `notebooks/` → separa lógica reutilizable de la narrativa y de la UI.
- **Idioma:** narrativa/EDA/app en español (audiencia = Gerencia); código y nombres en inglés (convención). (Default a confirmar.)
- **Alineación con el material de cátedra** (revisado: decks de árboles, RF, métricas, EDA, minería, feature engineering + prácticas). Se replica el molde del profe: framework **CRISP-DM**; `DecisionTreeClassifier` **podado** (max_depth/min_samples_leaf) + `RandomForestClassifier` **sin podar**; desbalance vía **muestreo estratificado + StratifiedKFold + métricas correctas** (no `class_weight`/SMOTE como mecanismo principal); evaluación con `classification_report` + `ConfusionMatrixDisplay` + ROC-AUC (PR-AUC opcional, no visto en clase); **One-Hot** (`get_dummies(drop_first=True)`/`OneHotEncoder`), nunca Label Encoding en nominales; binning de edad en grupos etarios; sin escalado para modelos de árbol; razonamiento explícito **FN vs FP** (costo clínico); `feature_importances_` y `plot_tree` para interpretabilidad; estilo de notebook con `target_name` vía `.map()`, nombres `X/y/X_train`, interpretaciones en Markdown.
- **Divergencias deliberadas (superadoras, no contradicen a la cátedra):** `ColumnTransformer`/`Pipeline` persistido con el modelo (el molde usa `get_dummies` suelto, pero la app exige preprocesado sin fuga y reproducible en inferencia) y `stratify=y` en el split (el molde no estratifica; se agrega por el desbalance y se justifica).

## Risks / Trade-offs

- **Python 3.14 sin wheels para pandas/sklearn/streamlit** → Mitigación: venv dedicado y, si falla, fallback a venv 3.12 documentado en el README.
- **Archivo de clima de 437 MB** → Mitigación: lectura por chunks filtrando A612, cachear un CSV diario chico; ambos crudos grandes al `.gitignore`.
- **`SMS_received` está confundido con el lead time** (se envía más a turnos con mayor anticipación) → Mitigación: documentarlo en el storytelling para no inducir una conclusión causal errónea a la Gerencia.
- **Barrios de alta cardinalidad (~81)** → Mitigación: agrupar categorías raras antes de one-hot para evitar explosión de dimensionalidad.
- **Datos de 2016, una sola ciudad, pre-COVID** → Mitigación: slide/sección de limitaciones y mención de reentrenamiento (concept drift). Es académico; no afecta la validación de la hipótesis.
- **Ceremonia sk8 (sub-branches/PRs atómicos)** desproporcionada para un TP → Mitigación: un branch de feature + commits convencionales por módulo.

## Migration Plan

No aplica migración (proyecto nuevo). Orden de construcción sugerido: entorno/deps → data-pipeline → weather-integration → modeling → interactive-app → notebook/docs. Rollback = revertir commits del branch de feature.

## Open Questions

- Confirmar idioma definitivo de código/comentarios (default: docs español / código inglés).
- ¿El docente considera el cruce turnos + clima como "varias fuentes distintas" para la expectativa superadora? (pregunta ya anotada en el desarrollo del grupo).
- ¿Incluir historial de no-show previo del paciente como feature avanzada, asumiendo el costo de evitar fuga temporal?
