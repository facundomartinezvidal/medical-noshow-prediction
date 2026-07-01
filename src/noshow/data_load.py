"""Carga y validación del dataset de turnos médicos (`no-show-dataset.csv`).

Responsabilidades:
- Validar que el CSV tenga exactamente el esquema de 14 columnas esperado.
- Parsear las columnas de fecha a datetime.
- Normalizar la variable objetivo `No-show` ("Yes"/"No") a una columna
  binaria `no_show` (1/0), conservando además `target_name` al estilo de la
  cátedra (`no_show` / `show`) para uso en EDA y notebooks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from noshow import config

logger = logging.getLogger(__name__)

DATE_COLUMNS: list[str] = ["ScheduledDay", "AppointmentDay"]

# Mapeo de la etiqueta cruda "Yes"/"No" de No-show a la variable binaria
# no_show (1 = faltó, 0 = asistió).
_TARGET_MAP: dict[str, int] = {"Yes": 1, "No": 0}
_TARGET_NAME_MAP: dict[int, str] = {1: "no_show", 0: "show"}


def validate_schema(df: pd.DataFrame) -> None:
    """Valida que `df` contenga todas las columnas del esquema esperado.

    Parameters
    ----------
    df:
        DataFrame a validar (típicamente recién leído del CSV crudo).

    Raises
    ------
    ValueError
        Si falta alguna columna del esquema esperado, nombrando las
        columnas faltantes.
    """
    missing = [col for col in config.APPOINTMENTS_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(
            "El dataset de turnos no tiene el esquema esperado. "
            f"Faltan las columnas: {missing}"
        )


def _normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `no_show` (1/0) y `target_name` a partir de la columna cruda
    `No-show` ("Yes"/"No").
    """
    unexpected = set(df["No-show"].unique()) - set(_TARGET_MAP)
    if unexpected:
        raise ValueError(
            "La columna 'No-show' contiene valores inesperados "
            f"(se esperaba 'Yes'/'No'): {sorted(unexpected)}"
        )

    df["no_show"] = df["No-show"].map(_TARGET_MAP).astype("int64")
    df["target_name"] = df["no_show"].map(_TARGET_NAME_MAP)
    return df


def load_appointments(path: Path = config.RAW_APPOINTMENTS) -> pd.DataFrame:
    """Carga el dataset de turnos, valida su esquema y normaliza tipos.

    Parameters
    ----------
    path:
        Ruta al CSV crudo de turnos. Por defecto `config.RAW_APPOINTMENTS`.

    Returns
    -------
    pd.DataFrame
        DataFrame con las 14 columnas originales + `no_show` (int64, 1/0) y
        `target_name` (str), y `ScheduledDay`/`AppointmentDay` parseadas a
        datetime.

    Raises
    ------
    ValueError
        Si el CSV no contiene el esquema esperado.
    """
    df = pd.read_csv(path)
    validate_schema(df)

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col])

    df = _normalize_target(df)

    logger.info(
        "Turnos cargados: %d filas, %d no-show (%.1f%%)",
        len(df),
        int(df["no_show"].sum()),
        100 * df["no_show"].mean(),
    )
    return df


if __name__ == "__main__":  # pragma: no cover - verificación manual
    logging.basicConfig(level=logging.INFO)
    frame = load_appointments()
    print(f"shape={frame.shape}")
    print(f"no_show sum={int(frame['no_show'].sum())}")
