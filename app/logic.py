"""Lógica pura (sin UI) de la aplicación interactiva de no-show.

Se separa deliberadamente de `streamlit_app.py` para poder testear estas
funciones sin depender de un runtime de Streamlit. Ninguna función de este
módulo reimplementa la ingeniería de características ni el preprocesado:
arma una fila/lote de turnos con las columnas CRUDAS del dataset original
y reutiliza `noshow.weather.merge_weather` + `noshow.features.build_features`
(el mismo camino que usó `noshow.train`) antes de scorear con el pipeline
persistido (`noshow.predict`). Así se garantiza idéntica transformación
entre entrenamiento e inferencia, sin fuga de datos ni recodificación manual.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from noshow import config
from noshow.features import build_features
from noshow.predict import predict_appointment, predict_batch, recommend_action
from noshow.weather import merge_weather

# --- Columnas crudas ---------------------------------------------------

# Columnas del dataset original necesarias para reproducir el camino de
# `merge_weather` + `build_features` (no se requieren PatientId/AppointmentID
# ni el target `No-show` para scorear: ninguna de las dos funciones los usa).
RAW_APPOINTMENT_COLUMNS: list[str] = [
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
]

# --- Supuesto de negocio: duración promedio de un turno -------------------

# No hay un dataset de duraciones reales; se asume un turno ambulatorio
# promedio de 30 minutos para traducir "no-shows evitados" (turnos de alto
# riesgo, ponderados por su propia probabilidad) en horas-profesional
# recuperables, que junto con `config.COST_HOUR` estiman el valor de negocio
# de accionar sobre la agenda del día.
DEFAULT_APPOINTMENT_DURATION_HOURS: float = 0.5


# --- Turno individual ----------------------------------------------------


def build_appointment_row(
    *,
    age: int,
    gender: str,
    neighbourhood: str,
    scheduled_day: dt.date | dt.datetime,
    appointment_day: dt.date | dt.datetime,
    sms_received: bool,
    hipertension: bool,
    diabetes: bool,
    alcoholism: bool,
    handcap: bool,
    scholarship: bool,
) -> pd.DataFrame:
    """Arma un DataFrame crudo de UN turno con las columnas del dataset
    original (`RAW_APPOINTMENT_COLUMNS`), listo para pasar por
    `merge_weather` + `build_features`.

    Valida edad no negativa y que el turno no sea anterior al agendamiento
    (`lead_time_days >= 0`), condiciones que de otro modo `build_features`
    descartaría silenciosamente (`clean_invalid_records`), dejando el
    turno sin score.
    """
    if age < 0:
        raise ValueError("La edad no puede ser negativa.")

    scheduled_ts = pd.Timestamp(scheduled_day)
    appointment_ts = pd.Timestamp(appointment_day)
    if appointment_ts.normalize() < scheduled_ts.normalize():
        raise ValueError(
            "La fecha del turno no puede ser anterior a la fecha de agendamiento."
        )

    row = {
        "Gender": gender,
        "ScheduledDay": scheduled_ts,
        "AppointmentDay": appointment_ts,
        "Age": int(age),
        "Neighbourhood": neighbourhood,
        "Scholarship": int(bool(scholarship)),
        "Hipertension": int(bool(hipertension)),
        "Diabetes": int(bool(diabetes)),
        "Alcoholism": int(bool(alcoholism)),
        "Handcap": int(bool(handcap)),
        "SMS_received": int(bool(sms_received)),
    }
    return pd.DataFrame([row], columns=RAW_APPOINTMENT_COLUMNS)


def lookup_weather_for_date(weather_daily_df: pd.DataFrame, date: dt.date) -> dict | None:
    """Busca el clima cacheado (`noshow.weather.load_weather_daily`) para
    `date`. Devuelve `None` si no hay dato para esa fecha (fuera del rango
    cubierto por la estación), en cuyo caso la app debe permitir el
    ingreso manual.
    """
    target = pd.Timestamp(date).normalize()
    dates = pd.to_datetime(weather_daily_df["date"]).dt.normalize()
    match = weather_daily_df.loc[dates == target]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "precipitation_mm": float(row["precipitation_mm"]),
        "is_rainy": int(row["is_rainy"]),
        "temp_mean": float(row["temp_mean"]),
    }


def apply_weather_override(
    merged_row: pd.DataFrame,
    *,
    precipitation_mm: float | None = None,
    is_rainy: bool | None = None,
    temp_mean: float | None = None,
) -> pd.DataFrame:
    """Sobrescribe las columnas climáticas de `merged_row` (salida de
    `merge_weather`) con valores ingresados manualmente, para el caso en
    que la fecha del turno no tenga clima cacheado disponible.
    """
    merged_row = merged_row.copy()
    if precipitation_mm is not None:
        merged_row["precipitation_mm"] = float(precipitation_mm)
    if is_rainy is not None:
        merged_row["is_rainy"] = int(bool(is_rainy))
    if temp_mean is not None:
        merged_row["temp_mean"] = float(temp_mean)
    return merged_row


def score_single_appointment(
    model,
    weather_daily_df: pd.DataFrame,
    raw_row: pd.DataFrame,
    weather_override: dict | None = None,
) -> tuple[float, str]:
    """Scorea UN turno crudo (`build_appointment_row`) con el pipeline
    persistido, reutilizando `merge_weather` + `build_features` (mismo
    camino que el entrenamiento) antes de `predict_appointment`.

    Returns
    -------
    tuple[float, str]
        `(probabilidad_no_show, accion_recomendada)`.
    """
    merged = merge_weather(raw_row, weather_daily_df)
    if weather_override:
        merged = apply_weather_override(merged, **weather_override)

    features = build_features(merged)
    if features.empty:
        raise ValueError(
            "El turno ingresado no es válido (edad o fecha de turno "
            "anteriores a lo permitido)."
        )

    proba = predict_appointment(model, features.iloc[0].to_dict())
    action = recommend_action(proba)
    return proba, action


def risk_band(proba: float) -> str:
    """Traduce una probabilidad de no-show en su banda de riesgo
    ("bajo"/"medio"/"alto"), usando los mismos umbrales que
    `noshow.predict.recommend_action` (`config.RISK_LOW`/`RISK_HIGH`).
    """
    if proba < config.RISK_LOW:
        return "bajo"
    if proba < config.RISK_HIGH:
        return "medio"
    return "alto"


RISK_BAND_COLORS: dict[str, str] = {"bajo": "green", "medio": "orange", "alto": "red"}


# --- Modo lote (agenda del día) -------------------------------------------


def get_categorical_options(model) -> dict[str, list[str]]:
    """Introspecciona el `OneHotEncoder` ya ajustado dentro del pipeline
    persistido para recuperar, sin reimplementar la ingeniería de
    características, las categorías que el modelo reconoce por columna
    categórica (`Gender`, `age_group`, `neighbourhood_grouped`).

    Usado para poblar los selectbox de la app con exactamente las
    categorías vistas en entrenamiento (incluida la agrupación "OTHER" de
    barrios de baja frecuencia).
    """
    preprocessor = model.named_steps["preprocessor"]
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            onehot = transformer.named_steps["onehot"]
            return {col: list(cats) for col, cats in zip(columns, onehot.categories_)}
    return {}


def score_batch(
    model, weather_daily_df: pd.DataFrame, raw_df: pd.DataFrame
) -> pd.DataFrame:
    """Scorea un lote de turnos crudos (ej. la agenda del día subida como
    CSV), reutilizando `merge_weather` + `build_features` + `predict_batch`.

    Devuelve el DataFrame resultante de `predict_batch` (ordenado de mayor
    a menor riesgo, con `no_show_proba`) más la columna
    `accion_recomendada` por turno.
    """
    missing = [c for c in RAW_APPOINTMENT_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Al archivo le faltan columnas requeridas: {missing}. "
            f"Se esperan las columnas: {RAW_APPOINTMENT_COLUMNS}."
        )

    merged = merge_weather(raw_df, weather_daily_df)
    features = build_features(merged)
    if features.empty:
        raise ValueError(
            "Ningún turno del archivo es válido (edades o fechas de turno "
            "fuera de rango)."
        )

    scored = predict_batch(model, features)
    scored["accion_recomendada"] = scored["no_show_proba"].apply(recommend_action)
    return scored


def sample_batch_csv_bytes() -> bytes:
    """Genera un CSV de ejemplo (2 turnos) con el formato esperado por
    `score_batch`, para que la app lo ofrezca como plantilla descargable.
    """
    sample = pd.DataFrame(
        {
            "Gender": ["F", "M"],
            "ScheduledDay": ["2016-04-29T08:00:00Z", "2016-04-29T09:15:00Z"],
            "AppointmentDay": ["2016-05-02T00:00:00Z", "2016-05-10T00:00:00Z"],
            "Age": [34, 62],
            "Neighbourhood": ["JARDIM DA PENHA", "MATA DA PRAIA"],
            "Scholarship": [0, 1],
            "Hipertension": [0, 1],
            "Diabetes": [0, 0],
            "Alcoholism": [0, 0],
            "Handcap": [0, 0],
            "SMS_received": [1, 0],
        },
        columns=RAW_APPOINTMENT_COLUMNS,
    )
    return sample.to_csv(index=False).encode("utf-8")


def estimate_business_value(
    scored_df: pd.DataFrame,
    *,
    risk_low: float = config.RISK_LOW,
    risk_high: float = config.RISK_HIGH,
    cost_hour: float = config.COST_HOUR,
    duration_hours: float = DEFAULT_APPOINTMENT_DURATION_HOURS,
) -> dict:
    """Estima el valor de negocio de accionar sobre una agenda ya scoreada
    (salida de `score_batch`/`predict_batch`, con columna `no_show_proba`).

    Los turnos de alto riesgo (`proba >= risk_high`) son los candidatos a
    intervención de mayor intensidad (llamado + sobreturno). Se usa la
    suma de sus probabilidades como el número ESPERADO de no-shows entre
    ellos (cada uno es una variable Bernoulli independiente de parámetro
    `proba`), lo que a su vez estima:

    - `sobreturnos_sugeridos`: cuántos sobreturnos controlados conviene
      programar para cubrir esas ausencias esperadas.
    - `horas_profesional_recuperables`: esas ausencias esperadas evitadas,
      traducidas a horas de agenda (asumiendo `duration_hours` por turno).
    - `valor_recuperado_ars`: esas horas valuadas a `cost_hour`.

    Raises
    ------
    ValueError
        Si `scored_df` no tiene la columna `no_show_proba` (ver
        `noshow.predict.predict_batch`).
    """
    if "no_show_proba" not in scored_df.columns:
        raise ValueError(
            "scored_df debe tener la columna 'no_show_proba' "
            "(ver noshow.predict.predict_batch)."
        )

    proba = scored_df["no_show_proba"]
    alto_mask = proba >= risk_high
    medio_mask = (proba >= risk_low) & ~alto_mask

    expected_no_shows_alto = float(proba[alto_mask].sum())
    horas_recuperables = expected_no_shows_alto * duration_hours
    valor_recuperado_ars = horas_recuperables * cost_hour

    return {
        "n_total": int(len(scored_df)),
        "n_alto_riesgo": int(alto_mask.sum()),
        "n_riesgo_medio": int(medio_mask.sum()),
        "n_bajo_riesgo": int((~alto_mask & ~medio_mask).sum()),
        "no_shows_esperados_alto_riesgo": round(expected_no_shows_alto, 2),
        "sobreturnos_sugeridos": int(round(expected_no_shows_alto)),
        "horas_profesional_recuperables": round(horas_recuperables, 2),
        "valor_recuperado_ars": round(valor_recuperado_ars, 2),
    }
