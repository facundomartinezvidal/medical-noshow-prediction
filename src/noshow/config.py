"""Configuración central del proyecto: rutas, esquema esperado y constantes
de negocio.

Todas las rutas se resuelven de forma relativa a la raíz del repositorio para
que el paquete funcione sin importar desde qué directorio se invoque.
"""

from __future__ import annotations

from pathlib import Path

# --- Rutas -------------------------------------------------------------

# Raíz del repositorio: src/noshow/config.py -> src/noshow -> src -> <root>
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
EXTERNAL_DIR: Path = DATA_DIR / "external"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"

RAW_APPOINTMENTS: Path = RAW_DIR / "no-show-dataset.csv"
RAW_WEATHER: Path = RAW_DIR / "weather-dataset.csv"

WEATHER_DAILY_CACHE: Path = EXTERNAL_DIR / "weather_daily_a612.csv"
PROCESSED_DATASET: Path = PROCESSED_DIR / "appointments_processed.csv"

# --- Clima ---------------------------------------------------------------

# Estación INMET de Vitória (ES), ciudad de origen del dataset de turnos.
WEATHER_STATION: str = "A612"

# --- Esquema esperado del dataset de turnos --------------------------------

APPOINTMENTS_SCHEMA: list[str] = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
]

# --- Umbrales de riesgo de negocio -----------------------------------------

# Bandas de probabilidad de no-show usadas por la app para recomendar acción:
# [0, RISK_LOW) -> bajo riesgo (sin acción especial)
# [RISK_LOW, RISK_HIGH) -> riesgo medio (recordatorio dirigido)
# [RISK_HIGH, 1] -> riesgo alto (recordatorio + sobreturno)
#
# Recalibrados a partir de la distribución real de `predict_proba` del
# Random Forest ganador (sin `class_weight`) sobre el dataset procesado
# completo (110.521 turnos, tasa base de no-show ≈ 20,2 %). Esa
# probabilidad sale comprimida (rango observado ≈ [0.01, 0.54]; p50≈0.23,
# p90≈0.345): con los umbrales anteriores (0.3/0.6) el 74,7 % de los
# turnos caía en "bajo" y la banda "alto" era INALCANZABLE (0.6 supera el
# máximo observado), por lo que la acción agresiva (llamado + sobreturno)
# nunca se disparaba y la app no separaba riesgo de verdad.
#
# - RISK_LOW = 0.28 (≈ percentil 66 de la distribución observada): deja
#   ~34 % de los turnos con alguna acción (medio + alto), en línea con el
#   objetivo de negocio de accionar sobre el tercio más riesgoso sin
#   saturar de recordatorios a toda la agenda.
# - RISK_HIGH = 0.345 (≈ percentil 90, el decil de mayor riesgo real que
#   produce el modelo): así el ~10 % de turnos con score más alto recibe
#   la acción agresiva, banda que antes nunca se activaba.
#
# Si se reentrena el modelo con datos, hiperparámetros o `class_weight`
# distintos, la distribución de `predict_proba` puede desplazarse y estos
# umbrales deberían recalcularse contra los nuevos percentiles.
RISK_LOW: float = 0.28
RISK_HIGH: float = 0.345

# Costo estimado de una hora-profesional ociosa (ARS), usado para traducir
# el no-show evitado en valor de negocio en la app.
COST_HOUR: float = 15000.0

# Umbral mínimo de frecuencia relativa para conservar un barrio como propia
# categoría en el feature engineering; por debajo se agrupa en "OTHER".
NEIGHBOURHOOD_MIN_FREQ: float = 0.01
