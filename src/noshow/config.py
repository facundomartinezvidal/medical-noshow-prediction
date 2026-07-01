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
RISK_LOW: float = 0.3
RISK_HIGH: float = 0.6

# Costo estimado de una hora-profesional ociosa (ARS), usado para traducir
# el no-show evitado en valor de negocio en la app.
COST_HOUR: float = 15000.0

# Umbral mínimo de frecuencia relativa para conservar un barrio como propia
# categoría en el feature engineering; por debajo se agrupa en "OTHER".
NEIGHBOURHOOD_MIN_FREQ: float = 0.01
