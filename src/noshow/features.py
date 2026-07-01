"""Ingeniería de características para el dataset de turnos.

Deriva las variables predictoras a partir de las columnas crudas: lead time,
día de la semana/mes del turno, grupos etarios (binning), indicadores de
comorbilidad y agrupamiento de barrios de baja frecuencia. También expone
`build_processed_dataset`, que orquesta carga -> cruce de clima -> features y
cachea el resultado en `data/processed/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from noshow import config
from noshow.data_load import load_appointments
from noshow.weather import load_weather_daily, merge_weather

logger = logging.getLogger(__name__)

AGE_BIN_EDGES: list[float] = [-1, 17, 64, np.inf]
AGE_BIN_LABELS: list[str] = ["menor", "adulto", "adulto_mayor"]

COMORBIDITY_COLUMNS: list[str] = ["Hipertension", "Diabetes", "Alcoholism"]


def compute_lead_time_days(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `lead_time_days` = días entre `ScheduledDay` y `AppointmentDay`.

    Ambas fechas se normalizan a medianoche antes de restar, para que la
    diferencia refleje días calendario y no se vea afectada por la hora en
    que se agendó el turno.
    """
    df = df.copy()
    scheduled = pd.to_datetime(df["ScheduledDay"]).dt.normalize()
    appointment = pd.to_datetime(df["AppointmentDay"]).dt.normalize()
    df["lead_time_days"] = (appointment - scheduled).dt.days
    return df


def clean_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta registros con `Age` negativa o `lead_time_days` negativo.

    Requiere que `lead_time_days` ya haya sido calculado (ver
    `compute_lead_time_days`). Deja constancia en el log de cuántos
    registros se descartaron por cada motivo.
    """
    df = df.copy()

    negative_age_mask = df["Age"] < 0
    n_negative_age = int(negative_age_mask.sum())
    if n_negative_age:
        logger.info(
            "Descartando %d registro(s) con Age negativa", n_negative_age
        )
        df = df.loc[~negative_age_mask]

    negative_lead_mask = df["lead_time_days"] < 0
    n_negative_lead = int(negative_lead_mask.sum())
    if n_negative_lead:
        logger.info(
            "Descartando %d registro(s) con lead_time_days negativo",
            n_negative_lead,
        )
        df = df.loc[~negative_lead_mask]

    return df.reset_index(drop=True)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `appointment_dow` (0=lunes), `appointment_month` y `same_day`."""
    df = df.copy()
    appointment = pd.to_datetime(df["AppointmentDay"])
    df["appointment_dow"] = appointment.dt.dayofweek.astype("int64")
    df["appointment_month"] = appointment.dt.month.astype("int64")
    df["same_day"] = (df["lead_time_days"] == 0).astype("int64")
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza `Age` en grupos etarios: menor (0-17) / adulto (18-64) /
    adulto_mayor (65+).
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["Age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS
    ).astype(str)
    return df


def add_comorbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `comorbidity_count` y `has_comorbidity` a partir de
    Hipertension/Diabetes/Alcoholism/Handcap.
    """
    df = df.copy()
    comorbidity_count = df[COMORBIDITY_COLUMNS].sum(axis=1)
    comorbidity_count = comorbidity_count + (df["Handcap"] > 0).astype("int64")
    df["comorbidity_count"] = comorbidity_count.astype("int64")
    df["has_comorbidity"] = (df["comorbidity_count"] > 0).astype("int64")
    return df


def group_rare_neighbourhoods(
    df: pd.DataFrame, min_freq: float = config.NEIGHBOURHOOD_MIN_FREQ
) -> pd.DataFrame:
    """Agrupa en "OTHER" los barrios cuya frecuencia relativa es menor a
    `min_freq`, reduciendo la cardinalidad de `Neighbourhood` antes del
    one-hot encoding.
    """
    df = df.copy()
    freqs = df["Neighbourhood"].value_counts(normalize=True)
    rare = freqs[freqs < min_freq].index
    df["neighbourhood_grouped"] = df["Neighbourhood"].where(
        ~df["Neighbourhood"].isin(rare), "OTHER"
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orquesta la ingeniería de características sobre un DataFrame de
    turnos (ya cargado y, opcionalmente, con clima cruzado).

    Pasos: lead time -> limpieza de inválidos -> features de fecha -> grupo
    etario -> comorbilidades -> agrupamiento de barrios raros.
    """
    df = compute_lead_time_days(df)
    df = clean_invalid_records(df)
    df = add_date_features(df)
    df = add_age_group(df)
    df = add_comorbidity_features(df)
    df = group_rare_neighbourhoods(df)
    return df


def build_processed_dataset(
    cache: Path = config.PROCESSED_DATASET, use_cache: bool = True
) -> pd.DataFrame:
    """Orquesta el pipeline completo: carga de turnos -> cruce de clima ->
    ingeniería de características, con cacheo opcional en
    `data/processed/`.
    """
    if use_cache and cache.exists():
        logger.info("Leyendo dataset procesado desde caché: %s", cache)
        return pd.read_csv(cache, parse_dates=["ScheduledDay", "AppointmentDay"])

    appointments = load_appointments()
    weather_daily = load_weather_daily()
    merged = merge_weather(appointments, weather_daily)
    processed = build_features(merged)

    cache.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(cache, index=False)
    logger.info("Dataset procesado cacheado en %s (%d filas)", cache, len(processed))

    return processed


if __name__ == "__main__":  # pragma: no cover - verificación manual
    logging.basicConfig(level=logging.INFO)
    result = build_processed_dataset()
    print(f"processed shape={result.shape}")
