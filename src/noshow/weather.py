"""Integración de la fuente secundaria de clima (INMET, estación A612 —
Vitória, ES) con el dataset de turnos.

El CSV crudo de INMET (`weather-dataset.csv`) pesa ~437 MB y contiene 529
estaciones a granularidad horaria. Para no cargarlo completo en memoria, se
lee por chunks, filtrando únicamente la estación de interés antes de
concatenar, y se agrega horario -> diario. El resultado (chico) se cachea en
`data/external/` para evitar re-procesar el archivo grande en corridas
posteriores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from noshow import config

logger = logging.getLogger(__name__)

# Columnas del CSV crudo de INMET que efectivamente se usan. Restringir las
# columnas leídas (`usecols`) reduce fuertemente el uso de memoria por chunk.
_COL_DATE = "DATA (YYYY-MM-DD)"
_COL_STATION = "ESTACAO"
_COL_PRECIP = "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"
_COL_TEMP_MEAN = "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"
_COL_TEMP_MAX = "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)"
_COL_TEMP_MIN = "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)"
_COL_HUMIDITY = "UMIDADE RELATIVA DO AR, HORARIA (%)"

RAW_WEATHER_USECOLS: list[str] = [
    _COL_DATE,
    _COL_STATION,
    _COL_PRECIP,
    _COL_TEMP_MEAN,
    _COL_TEMP_MAX,
    _COL_TEMP_MIN,
    _COL_HUMIDITY,
]

DEFAULT_CHUNKSIZE: int = 200_000

# Columnas del dataset diario ya agregado y con nombres normalizados (PT -> en).
WEATHER_DAILY_COLUMNS: list[str] = [
    "date",
    "precipitation_mm",
    "temp_max",
    "temp_min",
    "temp_mean",
    "humidity_mean",
    "is_rainy",
]


def _read_and_filter_station(
    path: Path, station: str, chunksize: int
) -> pd.DataFrame:
    """Lee `path` por chunks y concatena únicamente las filas de `station`.

    Nunca materializa el archivo completo en memoria: cada chunk se filtra
    antes de acumularse.
    """
    filtered_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=RAW_WEATHER_USECOLS,
        chunksize=chunksize,
    ):
        station_rows = chunk.loc[chunk[_COL_STATION] == station]
        if not station_rows.empty:
            filtered_chunks.append(station_rows)

    if not filtered_chunks:
        raise ValueError(
            f"No se encontraron registros para la estación '{station}' en {path}"
        )

    return pd.concat(filtered_chunks, ignore_index=True)


def _aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Agrega registros horarios de una estación a granularidad diaria."""
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly[_COL_DATE]).dt.date

    daily = hourly.groupby("date").agg(
        precipitation_mm=(_COL_PRECIP, "sum"),
        temp_max=(_COL_TEMP_MAX, "max"),
        temp_min=(_COL_TEMP_MIN, "min"),
        temp_mean=(_COL_TEMP_MEAN, "mean"),
        humidity_mean=(_COL_HUMIDITY, "mean"),
    )
    daily["is_rainy"] = (daily["precipitation_mm"] > 0).astype("int64")
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily[WEATHER_DAILY_COLUMNS]


def load_weather_daily(
    path: Path = config.RAW_WEATHER,
    station: str = config.WEATHER_STATION,
    cache: Path = config.WEATHER_DAILY_CACHE,
    chunksize: int = DEFAULT_CHUNKSIZE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Devuelve el clima diario agregado de `station`.

    Si `cache` ya existe (y `use_cache=True`), se lee directamente de ahí sin
    tocar el CSV crudo de 437 MB. Caso contrario, se procesa `path` por
    chunks, filtrando la estación y agregando horario -> diario, y se
    persiste el resultado en `cache`.

    Parameters
    ----------
    path:
        Ruta al CSV crudo horario de INMET.
    station:
        Código de estación a filtrar (ej. "A612" para Vitória, ES).
    cache:
        Ruta del CSV diario cacheado (chico), reutilizable entre corridas.
    chunksize:
        Cantidad de filas por chunk al leer el CSV crudo.
    use_cache:
        Si es False, fuerza el reprocesamiento del CSV crudo aunque el
        caché exista.

    Returns
    -------
    pd.DataFrame
        Columnas: `date`, `precipitation_mm`, `temp_max`, `temp_min`,
        `temp_mean`, `humidity_mean`, `is_rainy`.
    """
    if use_cache and cache.exists():
        logger.info("Leyendo clima diario desde caché: %s", cache)
        daily = pd.read_csv(cache, parse_dates=["date"])
        return daily[WEATHER_DAILY_COLUMNS]

    logger.info(
        "Procesando clima crudo por chunks (chunksize=%d) desde %s", chunksize, path
    )
    hourly = _read_and_filter_station(path, station, chunksize)
    daily = _aggregate_daily(hourly)

    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache, index=False)
    logger.info("Clima diario cacheado en %s (%d filas)", cache, len(daily))

    return daily


def merge_weather(
    appointments_df: pd.DataFrame, weather_daily_df: pd.DataFrame
) -> pd.DataFrame:
    """Une el clima diario a los turnos por la fecha de `AppointmentDay`.

    Los turnos cuya fecha no tiene clima disponible (fuera del rango
    cubierto por la estación) quedan con las columnas climáticas en NaN y
    reciben la marca `weather_missing=1` en lugar de romper el pipeline.
    """
    df = appointments_df.copy()
    appointment_day = pd.to_datetime(df["AppointmentDay"])
    if appointment_day.dt.tz is not None:
        # Las fechas de turnos suelen venir con sufijo "Z" (UTC). Se
        # descarta la zona horaria para poder cruzarlas con el clima
        # (naive, granularidad diaria) sin ambigüedad.
        appointment_day = appointment_day.dt.tz_localize(None)
    df["_appointment_date"] = appointment_day.dt.normalize()

    weather = weather_daily_df.copy()
    weather_date = pd.to_datetime(weather["date"])
    if weather_date.dt.tz is not None:
        weather_date = weather_date.dt.tz_localize(None)
    weather["date"] = weather_date.dt.normalize()

    merged = df.merge(
        weather,
        left_on="_appointment_date",
        right_on="date",
        how="left",
    )
    merged = merged.drop(columns=["_appointment_date", "date"])

    weather_cols = [c for c in WEATHER_DAILY_COLUMNS if c != "date"]
    merged["weather_missing"] = merged[weather_cols[0]].isna().astype("int64")

    n_missing = int(merged["weather_missing"].sum())
    if n_missing:
        logger.info(
            "%d turnos sin clima disponible para su fecha (weather_missing=1)",
            n_missing,
        )

    # Imputación explícita: precipitación/lluvia ausente -> 0 (no se registró
    # lluvia); temperatura/humedad ausente -> media de la serie disponible.
    merged["precipitation_mm"] = merged["precipitation_mm"].fillna(0.0)
    merged["is_rainy"] = merged["is_rainy"].fillna(0).astype("int64")
    for col in ("temp_max", "temp_min", "temp_mean", "humidity_mean"):
        if merged[col].notna().any():
            merged[col] = merged[col].fillna(merged[col].mean())

    return merged


if __name__ == "__main__":  # pragma: no cover - verificación manual
    logging.basicConfig(level=logging.INFO)
    daily_df = load_weather_daily()
    print(f"weather daily shape={daily_df.shape}")
    print(daily_df.head())
