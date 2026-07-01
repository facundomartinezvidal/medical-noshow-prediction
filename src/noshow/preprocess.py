"""Preprocesado sin fuga de datos para el modelado de no-show.

Encapsula la codificación de categóricas nominales (One-Hot) y la
imputación simple en un `ColumnTransformer` de scikit-learn, para que sea
ajustado únicamente sobre el conjunto de entrenamiento y persistido junto
con el modelo (misma transformación en entrenamiento e inferencia, sin
fuga de datos).

Los modelos de árbol (Decision Tree / Random Forest) no requieren
escalado de variables numéricas, conforme a lo enseñado en la cátedra:
las numéricas se pasan sin transformar (solo se imputan valores
faltantes, si los hubiera).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COL: str = "no_show"

# Identificadores, fechas crudas y columnas derivadas del target: nunca
# deben usarse como feature (no aportan señal generalizable o directamente
# filtran el target).
NON_FEATURE_COLUMNS: list[str] = [
    "PatientId",
    "AppointmentID",
    "ScheduledDay",
    "AppointmentDay",
    "Neighbourhood",  # reemplazada por `neighbourhood_grouped` (menor cardinalidad)
    "No-show",
    "target_name",
    TARGET_COL,
]

# Categóricas nominales: se codifican con One-Hot para no inyectar un
# orden artificial (conforme a la cátedra, nunca Label Encoding aquí).
CATEGORICAL_COLUMNS: list[str] = ["Gender", "age_group", "neighbourhood_grouped"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa las columnas de `df` en numéricas/binarias y categóricas
    nominales, excluyendo identificadores, fechas crudas y el target.

    Returns
    -------
    tuple[list[str], list[str]]
        `(numeric_cols, categorical_cols)`, ambas restringidas a las
        columnas efectivamente presentes en `df`.
    """
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    numeric_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and c not in categorical_cols
    ]
    return numeric_cols, categorical_cols


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa `df` en `X` (features) e `y` (target `no_show`, 1/0).

    `X` conserva únicamente las columnas devueltas por
    `get_feature_columns` (numéricas + categóricas), en ese orden.
    """
    numeric_cols, categorical_cols = get_feature_columns(df)
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype("int64")
    return X, y


def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str]
) -> ColumnTransformer:
    """Arma el `ColumnTransformer` de preprocesado.

    - Numéricas/binarias: imputación por mediana, SIN escalado (los
      árboles de decisión y Random Forest no lo requieren).
    - Categóricas nominales: imputación por moda + `OneHotEncoder`
      (`handle_unknown="ignore"` para no romper en inferencia ante una
      categoría nueva no vista en entrenamiento).

    El `ColumnTransformer` debe ajustarse únicamente sobre el conjunto de
    entrenamiento (ver `noshow.train`) para evitar fuga de datos.
    """
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
