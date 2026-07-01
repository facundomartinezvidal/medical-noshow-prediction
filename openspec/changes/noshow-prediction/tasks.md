## 1. Entorno y scaffolding

- [ ] 1.1 Crear estructura de carpetas: `data/{raw,external,processed}/`, `src/noshow/`, `app/`, `notebooks/`, `models/`, `reports/figures/`, `docs/`
- [ ] 1.2 Crear `.gitignore` (datos crudos grandes `*.csv` de raw, `.venv/`, `models/*.joblib`, `__pycache__/`, `.DS_Store`, artefactos openspec `opsx` si corresponde)
- [ ] 1.3 Crear `requirements.txt` (`pandas`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`, `joblib`, `jupyter`)
- [ ] 1.4 Crear venv e instalar; si faltan wheels para Python 3.14, documentar fallback venv 3.12 en el README
- [ ] 1.5 `src/noshow/config.py` con paths, esquema esperado y constantes de negocio (umbrales de riesgo, costo hora-profesional)

## 2. data-pipeline

- [ ] 2.1 `data_load.py`: cargar CSV de turnos, validar 14 columnas, parsear fechas, normalizar target `No-show`→`no_show` (1/0)
- [ ] 2.2 Limpieza: edades negativas y lead times negativos, con conteo de registros afectados
- [ ] 2.3 `features.py`: `lead_time_days`, `appointment_dow`, `appointment_month`, `same_day`, bins de edad, comorbilidades, agrupado de barrios raros
- [ ] 2.4 Generar y cachear el dataset procesado en `data/processed/` de forma determinística

## 3. weather-integration

- [ ] 3.1 `weather.py`: leer `weather-dataset.csv` por chunks filtrando `ESTACAO=="A612"` (sin cargar 437 MB de golpe)
- [ ] 3.2 Mapear columnas PT→limpio; agregar horario→diario (precip suma, temp máx/mín/media, humedad media, `is_rainy`)
- [ ] 3.3 Cachear el clima diario chico en `data/external/`
- [ ] 3.4 Join del clima con los turnos por fecha de `AppointmentDay`, con manejo explícito de fechas sin clima

## 4. noshow-modeling

- [ ] 4.1 `preprocess.py`: `ColumnTransformer` (imputación, OHE de categóricas, escalado de numéricas) reutilizable
- [ ] 4.2 `train.py`: split estratificado train/test; entrenar DecisionTree y RandomForest con `class_weight="balanced"`
- [ ] 4.3 `evaluate.py`: precision, recall, F1, ROC-AUC, PR-AUC, matriz de confusión; comparativa DT vs RF; importancia de variables
- [ ] 4.4 Ajuste de umbral orientado a recall de la clase "falta"
- [ ] 4.5 Persistir modelo final (pipeline + clasificador) en `models/*.joblib` + metadata de métricas
- [ ] 4.6 `predict.py`: cargar modelo, scorear turno individual y lote, devolver `predict_proba`

## 5. interactive-app (Streamlit)

- [ ] 5.1 `app/streamlit_app.py` modo turno individual: inputs → P(no-show) con gauge/banda de riesgo, reutilizando el pipeline persistido
- [ ] 5.2 Mapear probabilidad → acción recomendada por bandas (bajo/medio/alto)
- [ ] 5.3 Modo lote: subir CSV de agenda → tabla rankeada por riesgo + acción por turno
- [ ] 5.4 Estimación de valor de negocio (horas-profesional recuperadas / sobreturnos sugeridos)

## 6. Notebook, visualización y docs

- [ ] 6.1 `notebooks/01_eda_modeling.ipynb`: EDA (lead time vs no-show, efecto SMS con caveat causal, edad, día de semana, clima) + storytelling en español
- [ ] 6.2 Exportar figuras clave a `reports/figures/` para la presentación
- [ ] 6.3 `docs/architecture.md`: diagrama de tubería en Mermaid
- [ ] 6.4 `README.md`: instalación, dónde poner los datos, cómo entrenar y cómo correr la app; sección de limitaciones (datos 2016, una ciudad, pre-COVID, concept drift)

## 7. Verificación end-to-end

- [ ] 7.1 `python -m noshow.data_load` valida esquema sin error
- [ ] 7.2 `python -m noshow.train` produce `models/*.joblib` + reporte de métricas (AUC/F1)
- [ ] 7.3 `streamlit run app/streamlit_app.py` levanta y devuelve P(no-show) + acción para un turno de prueba
- [ ] 7.4 El notebook corre de punta a punta y exporta figuras
