"""Entrenamiento de los modelos de clasificación de no-show.

Compara un Árbol de Decisión (podado con `max_depth`/`min_samples_leaf`) y
un Random Forest (sin podar) mediante validación cruzada estratificada
sobre el conjunto de entrenamiento, selecciona el modelo final por CV, y
reserva el hold-out de test para la evaluación final. El desbalance de
clases (~80/20) se aborda con `stratify=y` en el split y
`StratifiedKFold` en la validación cruzada, y con métricas robustas al
desbalance (F1 y ROC-AUC de la clase minoritaria), conforme a la
cátedra: `class_weight="balanced"` no se usa como mecanismo principal.

El pipeline ganador (preprocesado + clasificador), ajustado sobre el
conjunto de entrenamiento completo, se persiste en `models/model.joblib`
junto con `models/metrics.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from noshow import config
from noshow.preprocess import build_preprocessor, get_feature_columns, split_features_target

logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 5
CV_SCORING: list[str] = ["f1", "roc_auc"]

# Árbol de Decisión PODADO: se limita su complejidad para controlar el
# sobreajuste (conforme a la cátedra).
DT_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "min_samples_leaf": 50,
    "random_state": RANDOM_STATE,
}

# Random Forest SIN podar (el ensamble compensa el sobreajuste de árboles
# individuales; la cátedra no lo poda).
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MODEL_FACTORIES: dict[str, Any] = {
    "decision_tree": lambda: DecisionTreeClassifier(**DT_PARAMS),
    "random_forest": lambda: RandomForestClassifier(**RF_PARAMS),
}


def _build_pipeline(preprocessor, classifier) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Corre `StratifiedKFold` sobre `X_train`/`y_train` y agrega media y
    desvío estándar de F1 y ROC-AUC entre folds (no un único corte).
    """
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING)
    return {
        "f1_mean": float(scores["test_f1"].mean()),
        "f1_std": float(scores["test_f1"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
    }


def train_models(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Entrena Árbol de Decisión y Random Forest sobre `df`, los compara
    por validación cruzada estratificada en train, y evalúa ambos en el
    hold-out de test.

    Returns
    -------
    dict con, entre otras claves:
        - `best_model`: nombre del modelo elegido por CV (`decision_tree`
          o `random_forest`).
        - `cv_results`: media±std de F1/ROC-AUC por modelo (sobre train).
        - `test_metrics`: `classification_report` + `roc_auc` por modelo
          sobre el hold-out de test.
        - `hyperparameters`: hiperparámetros usados por cada modelo.
        - `pipeline`: el Pipeline (preprocesado + clasificador) del
          modelo ganador, ya ajustado sobre `X_train`.
        - `pipelines`: los Pipeline ajustados de AMBOS modelos.
        - `X_train`, `X_test`, `y_train`, `y_test`: la partición hold-out.
    """
    numeric_cols, categorical_cols = get_feature_columns(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    cv_results: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
    test_metrics: dict[str, dict[str, Any]] = {}

    for name, factory in MODEL_FACTORIES.items():
        # Un ColumnTransformer nuevo por modelo: cada Pipeline se ajusta de
        # forma independiente y exclusivamente sobre X_train (sin fuga).
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = _build_pipeline(preprocessor, factory())

        logger.info("Validación cruzada (%d folds) para %s", CV_SPLITS, name)
        cv_results[name] = _cross_validate_model(pipeline, X_train, y_train, random_state)

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics[name] = {
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

    # Selección del modelo final por desempeño de CV en train. Se usa
    # ROC-AUC (media entre folds) como criterio principal: a diferencia
    # de F1, no depende de fijar de antemano un umbral de 0.5 arbitrario
    # -que bajo ~80/20 de desbalance castiga a ambos modelos por igual y
    # vuelve la comparación de F1 ruidosa/casi degenerada- sino que mide
    # la calidad del ranking de probabilidades que después se ajusta con
    # el umbral de negocio (ver `noshow.evaluate.threshold_analysis`). El
    # hold-out de test queda reservado solo para la evaluación final,
    # nunca para elegir el modelo.
    best_name = max(cv_results, key=lambda name: cv_results[name]["roc_auc_mean"])

    logger.info(
        "Modelo elegido por CV: %s (F1=%.3f±%.3f, ROC-AUC=%.3f±%.3f)",
        best_name,
        cv_results[best_name]["f1_mean"],
        cv_results[best_name]["f1_std"],
        cv_results[best_name]["roc_auc_mean"],
        cv_results[best_name]["roc_auc_std"],
    )

    return {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "hyperparameters": {"decision_tree": DT_PARAMS, "random_forest": RF_PARAMS},
        "pipeline": fitted_pipelines[best_name],
        "pipelines": fitted_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def save_artifacts(
    result: dict[str, Any], models_dir: Path = config.MODELS_DIR
) -> tuple[Path, Path]:
    """Persiste el pipeline ganador (`model.joblib`) y las métricas
    (`metrics.json`) en `models_dir`.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    metrics_path = models_dir / "metrics.json"

    joblib.dump(result["pipeline"], model_path)

    metrics_payload = {
        "best_model": result["best_model"],
        "cv_results": result["cv_results"],
        "test_metrics": result["test_metrics"],
        "hyperparameters": result["hyperparameters"],
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Modelo persistido en %s", model_path)
    logger.info("Métricas persistidas en %s", metrics_path)
    return model_path, metrics_path


def _print_report(result: dict[str, Any]) -> None:
    print(f"Modelo elegido por validación cruzada: {result['best_model']}")
    print()
    print("Validación cruzada (train, 5-fold estratificado):")
    for name, cv in result["cv_results"].items():
        print(
            f"  {name}: F1={cv['f1_mean']:.3f}±{cv['f1_std']:.3f}  "
            f"ROC-AUC={cv['roc_auc_mean']:.3f}±{cv['roc_auc_std']:.3f}"
        )
    print()
    print("Evaluación en hold-out de test (clase no_show=1):")
    for name, metrics in result["test_metrics"].items():
        report = metrics["classification_report"]
        positive = report.get("1", {})
        print(
            f"  {name}: precision={positive.get('precision', float('nan')):.3f}  "
            f"recall={positive.get('recall', float('nan')):.3f}  "
            f"f1={positive.get('f1-score', float('nan')):.3f}  "
            f"roc_auc={metrics['roc_auc']:.3f}"
        )


def main() -> None:
    """Punto de entrada de `python -m noshow.train`: entrena sobre el
    dataset procesado real, persiste el modelo elegido y sus métricas, e
    imprime un reporte comparativo por consola.
    """
    logging.basicConfig(level=logging.INFO)

    from noshow.features import build_processed_dataset

    df = build_processed_dataset()
    logger.info("Dataset de modelado: %d filas, %d columnas", *df.shape)

    result = train_models(df)
    model_path, metrics_path = save_artifacts(result)

    _print_report(result)
    print()
    print(f"Modelo persistido en: {model_path}")
    print(f"Métricas persistidas en: {metrics_path}")


if __name__ == "__main__":  # pragma: no cover - verificación manual
    main()
