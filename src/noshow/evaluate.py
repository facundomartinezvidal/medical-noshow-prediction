"""Evaluación de los modelos de no-show: métricas, curvas y artefactos de
interpretabilidad (matriz de confusión, curva ROC, importancia de
variables del Random Forest, visualización del Árbol de Decisión podado).

El desbalance de clases (~80/20) hace que la accuracy sea una métrica
engañosa: se prioriza `classification_report` (precision/recall/F1 por
clase), la matriz de confusión y ROC-AUC. La matriz de confusión es
además el punto de partida para razonar el costo relativo de un falso
negativo (un no-show real que el modelo no detecta, y que se traduce en
una hora-profesional ociosa) frente a un falso positivo (un turno que sí
se iba a cumplir y recibe una intervención de más costo bajo, como un
recordatorio); ese costo asimétrico es el que justifica priorizar recall
de la clase `no_show` sobre precision al elegir el umbral de decisión.

Todas las figuras se generan con el backend no interactivo `Agg` de
matplotlib (sin display), pensado para correr en scripts/notebooks sin
entorno gráfico.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import export_text, plot_tree  # noqa: E402

from noshow import config  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES_DIR: Path = config.ROOT_DIR / "reports" / "figures"

# Umbrales de decisión explorados en el ajuste orientado a recall de la
# clase `no_show`. Incluye las bandas de negocio (config.RISK_LOW/HIGH).
DEFAULT_THRESHOLDS: list[float] = [
    0.1,
    0.2,
    config.RISK_LOW,
    0.4,
    0.5,
    config.RISK_HIGH,
    0.7,
    0.8,
]


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Evalúa `pipeline` (ya ajustado) sobre el hold-out de test.

    Returns
    -------
    dict con:
        - `classification_report`: dict (accuracy, precision/recall/F1 por
          clase).
        - `roc_auc`: `roc_auc_score` sobre `predict_proba`.
        - `confusion_matrix`: `np.ndarray` de forma (2, 2).
        - `roc_curve`: dict con `fpr`, `tpr`, `thresholds`.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    return {
        "classification_report": report,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
    }


def threshold_analysis(
    y_test: pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Calcula precision/recall/F1 de la clase `no_show` (positiva, 1)
    para cada umbral en `thresholds`.

    Pensado para el ajuste de umbral orientado a recall: bajar el umbral
    respecto del 0.5 por defecto aumenta el recall (menos falsos
    negativos) a costa de precision, lo cual suele preferirse dado el
    mayor costo clínico/operativo de un no-show no detectado frente al de
    una intervención de más sobre un turno que igual se iba a cumplir.
    """
    y_true = np.asarray(y_test)
    y_proba = np.asarray(y_proba)

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: np.ndarray, output_path: Path = FIGURES_DIR / "confusion_matrix.png"
) -> Path:
    """Exporta la matriz de confusión con `ConfusionMatrixDisplay`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["show", "no_show"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: Path = FIGURES_DIR / "roc_curve.png",
) -> Path:
    """Exporta la curva ROC junto a la diagonal de referencia (azar)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_feature_importances(
    pipeline: Pipeline,
    output_path: Path = FIGURES_DIR / "feature_importances.png",
    top_n: int = 20,
) -> Path:
    """Grafica `feature_importances_` del Random Forest en barras
    horizontales ordenadas (storytelling técnico de interpretabilidad).

    Requiere que `pipeline` tenga los pasos `"preprocessor"` (con
    `get_feature_names_out`) y `"classifier"` (con
    `feature_importances_`, ej. `RandomForestClassifier`).
    """
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "El clasificador no expone feature_importances_ "
            "(¿es un RandomForestClassifier?)"
        )

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = classifier.feature_importances_

    order = np.argsort(importances)[-top_n:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.3)))
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables (Random Forest)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_decision_tree(
    pipeline: Pipeline,
    output_path: Path = FIGURES_DIR / "decision_tree.png",
    max_depth: int | None = 3,
) -> Path:
    """Visualiza el Árbol de Decisión podado con `plot_tree`.

    `max_depth` limita la profundidad graficada (no la del árbol
    entrenado) para mantener la figura legible.
    """
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "tree_"):
        raise ValueError("El clasificador no es un DecisionTreeClassifier")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        classifier,
        feature_names=list(feature_names),
        class_names=["show", "no_show"],
        filled=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def export_decision_tree_text(pipeline: Pipeline) -> str:
    """Devuelve la representación textual del árbol podado
    (`export_text`), útil para incluir en el notebook/reporte sin generar
    una imagen.
    """
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "tree_"):
        raise ValueError("El clasificador no es un DecisionTreeClassifier")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return export_text(classifier, feature_names=list(feature_names))
