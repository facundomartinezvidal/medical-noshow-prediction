"""Scoring de turnos con el pipeline entrenado (preprocesado + clasificador).

Provee la interfaz que consume la app interactiva: cargar el modelo
persistido, predecir la probabilidad de no-show de un turno individual o
de un lote (`predict_proba`, nunca solo la etiqueta), y traducir esa
probabilidad en una recomendación de acción de negocio según las bandas
de riesgo definidas en `noshow.config` (`RISK_LOW`/`RISK_HIGH`).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from noshow import config

DEFAULT_MODEL_PATH: Path = config.MODELS_DIR / "model.joblib"


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Carga el pipeline (preprocesado + clasificador) persistido en
    `path` (por defecto `models/model.joblib`).
    """
    return joblib.load(path)


def predict_appointment(model: Pipeline, features: dict) -> float:
    """Predice la probabilidad de no-show de UN turno.

    Arma un DataFrame de una fila a partir de `features` (debe contener
    las columnas que el pipeline espera como input) y devuelve
    `predict_proba` para la clase positiva (`no_show=1`).

    Parameters
    ----------
    model:
        Pipeline (preprocesado + clasificador) ya entrenado.
    features:
        Diccionario `{columna: valor}` con las features de un turno.

    Returns
    -------
    float
        Probabilidad estimada de no-show, en [0, 1].
    """
    row = pd.DataFrame([features])
    proba = model.predict_proba(row)[:, 1]
    return float(proba[0])


def predict_batch(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Scorea un lote de turnos y agrega la columna `no_show_proba`.

    Devuelve una copia de `df` (no modifica el original) ordenada de
    mayor a menor riesgo, lista para el ranking que consume la app en
    modo lote.
    """
    result = df.copy()
    result["no_show_proba"] = model.predict_proba(df)[:, 1]
    result = result.sort_values("no_show_proba", ascending=False).reset_index(drop=True)
    return result


def recommend_action(proba: float) -> str:
    """Traduce una probabilidad de no-show en una acción de negocio.

    Bandas (definidas en `noshow.config`):
        - `[0, RISK_LOW)`: riesgo bajo -> "sin acción".
        - `[RISK_LOW, RISK_HIGH)`: riesgo medio -> "recordatorio SMS".
        - `[RISK_HIGH, 1]`: riesgo alto -> "llamado + sobreturno".
    """
    if proba < config.RISK_LOW:
        return "sin acción"
    if proba < config.RISK_HIGH:
        return "recordatorio SMS"
    return "llamado + sobreturno"
