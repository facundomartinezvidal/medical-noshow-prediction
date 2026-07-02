## 1. Entorno y scaffolding

- [x] 1.1 Crear estructura de carpetas: `data/{raw,external,processed}/`, `src/noshow/`, `app/`, `notebooks/`, `models/`, `reports/figures/`, `docs/`
- [x] 1.2 Crear `.gitignore` (datos crudos grandes `*.csv` de raw, `.venv/`, `models/*.joblib`, `__pycache__/`, `.DS_Store`, artefactos openspec `opsx` si corresponde)
- [x] 1.3 Crear `requirements.txt` (`pandas`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`, `joblib`, `jupyter`)
- [x] 1.4 Crear venv e instalar; si faltan wheels para Python 3.14, documentar fallback venv 3.12 en el README
- [x] 1.5 `src/noshow/config.py` con paths, esquema esperado y constantes de negocio (umbrales de riesgo, costo hora-profesional)

## 2. data-pipeline

- [x] 2.1 `data_load.py`: cargar CSV de turnos, validar 14 columnas, parsear fechas, normalizar target `No-show`→`no_show` (1/0)
- [x] 2.2 Limpieza: edades negativas y lead times negativos, con conteo de registros afectados
- [x] 2.3 `features.py`: `lead_time_days`, `appointment_dow`, `appointment_month`, `same_day`, bins de edad, comorbilidades, agrupado de barrios raros
- [x] 2.4 Generar y cachear el dataset procesado en `data/processed/` de forma determinística

## 3. weather-integration

- [x] 3.1 `weather.py`: leer `weather-dataset.csv` por chunks filtrando `ESTACAO=="A612"` (sin cargar 437 MB de golpe)
- [x] 3.2 Mapear columnas PT→limpio; agregar horario→diario (precip suma, temp máx/mín/media, humedad media, `is_rainy`)
- [x] 3.3 Cachear el clima diario chico en `data/external/`
- [x] 3.4 Join del clima con los turnos por fecha de `AppointmentDay`, con manejo explícito de fechas sin clima

## 4. noshow-modeling

- [x] 4.1 `preprocess.py`: `ColumnTransformer` reutilizable — One-Hot (`OneHotEncoder(handle_unknown="ignore")`) de categóricas nominales (género, barrio, grupo etario); sin escalado (DT/RF no lo requieren)
- [x] 4.2 `train.py`: hold-out `train_test_split(test_size=0.2, random_state=42, stratify=y)`; entrenar `DecisionTreeClassifier(max_depth, min_samples_leaf, random_state=42)` **podado** y `RandomForestClassifier(n_estimators, max_features, max_depth, random_state=42)` sin podar
- [x] 4.3 Validación cruzada `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre train (`cross_val_score`); comparar DT vs RF por media±std de F1/ROC-AUC; elegir modelo final con CV
- [x] 4.4 `evaluate.py`: `classification_report` + `ConfusionMatrixDisplay` + `roc_auc_score` (y curva ROC) sobre el hold-out; discutir FN vs FP (costo clínico); ajuste de umbral orientado a recall; (opcional) PR-AUC
- [x] 4.5 Interpretabilidad: `plot_tree`/`export_text` del árbol podado (mostrar train vs test = overfitting) y `feature_importances_` del RF en barras horizontales ordenadas
- [x] 4.6 Persistir modelo final (pipeline + clasificador) en `models/*.joblib` + metadata de métricas
- [x] 4.7 `predict.py`: cargar modelo, scorear turno individual y lote, devolver `predict_proba` (probabilidad promedio del bosque)

## 5. interactive-app (Streamlit)

- [x] 5.1 `app/streamlit_app.py` modo turno individual: inputs → P(no-show) con gauge/banda de riesgo, reutilizando el pipeline persistido
- [x] 5.2 Mapear probabilidad → acción recomendada por bandas (bajo/medio/alto)
- [x] 5.3 Modo lote: subir CSV de agenda → tabla rankeada por riesgo + acción por turno
- [x] 5.4 Estimación de valor de negocio (horas-profesional recuperadas / sobreturnos sugeridos)

## 6. Notebook, visualización y docs

- [x] 6.1 `notebooks/01_eda_modeling.ipynb`: estructurar según **CRISP-DM**; EDA estilo cátedra (`.head/.shape/.dtypes/.isnull().sum()/.describe()`, barra del target comentando el balance, `boxplot`/`histplot` por clase, matriz de correlación con heatmap); insights (lead time vs no-show, efecto SMS con caveat causal, edad, día de semana, clima)
- [x] 6.2 Estilo cátedra: `df` con `target`+`target_name` vía `.map()`; nombres `X, y, X_train...`; interpretaciones y conclusiones en celdas Markdown; storytelling en español
- [x] 6.3 Exportar figuras clave a `reports/figures/` para la presentación
- [x] 6.4 `docs/architecture.md`: diagrama de tubería en Mermaid
- [x] 6.5 `README.md`: instalación, dónde poner los datos, cómo entrenar y cómo correr la app; sección de limitaciones (datos 2016, una ciudad, pre-COVID, concept drift)

## 7. Verificación end-to-end

- [x] 7.1 `python -m noshow.data_load` valida esquema sin error
- [x] 7.2 `python -m noshow.train` produce `models/*.joblib` + reporte de métricas (AUC/F1)
- [x] 7.3 `streamlit run app/streamlit_app.py` levanta y devuelve P(no-show) + acción para un turno de prueba
- [x] 7.4 El notebook corre de punta a punta y exporta figuras
