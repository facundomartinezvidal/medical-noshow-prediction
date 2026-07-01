"""Tests de la capa de modelado (`noshow.preprocess`, `noshow.train`,
`noshow.evaluate`, `noshow.predict`).

Todos los tests usan un DataFrame sintético pequeño con las columnas del
dataset procesado (~80 filas, target desbalanceado ~80/20): NUNCA se lee
el dataset real de 110.521 filas dentro de un test unitario.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from noshow import config
from noshow.evaluate import evaluate_model, threshold_analysis
from noshow.predict import predict_appointment, predict_batch, recommend_action
from noshow.preprocess import (
    build_preprocessor,
    get_feature_columns,
    split_features_target,
)
from noshow.train import save_artifacts, train_models

# --------------------------------------------------------------------------
# Fixture: dataset procesado sintético
# --------------------------------------------------------------------------


def _make_processed_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Arma un DataFrame sintético con la forma del dataset procesado real
    (columnas crudas + features de `noshow.features` + clima), con un
    target `no_show` desbalanceado (~20 % positivos, similar al dataset
    real de ~20,2 %).
    """
    rng = np.random.default_rng(seed)

    n_pos = round(n * 0.2)
    n_neg = n - n_pos
    no_show = np.array([0] * n_neg + [1] * n_pos)
    rng.shuffle(no_show)

    genders = rng.choice(["F", "M"], size=n)
    age = rng.integers(0, 90, size=n)
    age_group = (
        pd.cut(age, bins=[-1, 17, 64, np.inf], labels=["menor", "adulto", "adulto_mayor"])
        .astype(str)
    )
    neighbourhood = rng.choice(["JARDIM DA PENHA", "MATA DA PRAIA", "GOIABEIRAS"], size=n)

    scheduled = pd.date_range("2016-04-20", periods=n, freq="h")
    lead_time_days = rng.integers(0, 10, size=n)
    appointment = pd.to_datetime(scheduled.normalize()) + pd.to_timedelta(
        lead_time_days, unit="D"
    )

    return pd.DataFrame(
        {
            "PatientId": np.arange(1, n + 1, dtype=float),
            "AppointmentID": np.arange(1000, 1000 + n),
            "Gender": genders,
            "ScheduledDay": scheduled,
            "AppointmentDay": appointment,
            "Age": age,
            "Neighbourhood": neighbourhood,
            "Scholarship": rng.integers(0, 2, size=n),
            "Hipertension": rng.integers(0, 2, size=n),
            "Diabetes": rng.integers(0, 2, size=n),
            "Alcoholism": rng.integers(0, 2, size=n),
            "Handcap": rng.integers(0, 2, size=n),
            "SMS_received": rng.integers(0, 2, size=n),
            "No-show": np.where(no_show == 1, "Yes", "No"),
            "no_show": no_show,
            "target_name": np.where(no_show == 1, "no_show", "show"),
            "lead_time_days": lead_time_days,
            "appointment_dow": appointment.dayofweek,
            "appointment_month": appointment.month,
            "same_day": (lead_time_days == 0).astype(int),
            "age_group": age_group,
            "comorbidity_count": rng.integers(0, 3, size=n),
            "has_comorbidity": rng.integers(0, 2, size=n),
            "neighbourhood_grouped": neighbourhood,
            "precipitation_mm": rng.uniform(0, 5, size=n),
            "temp_max": rng.uniform(25, 32, size=n),
            "temp_min": rng.uniform(18, 24, size=n),
            "temp_mean": rng.uniform(20, 28, size=n),
            "humidity_mean": rng.uniform(60, 90, size=n),
            "is_rainy": rng.integers(0, 2, size=n),
            "weather_missing": np.zeros(n, dtype=int),
        }
    )


@pytest.fixture()
def processed_df() -> pd.DataFrame:
    return _make_processed_df()


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_excludes_identifiers_dates_and_target(self, processed_df):
        numeric_cols, categorical_cols = get_feature_columns(processed_df)
        all_feature_cols = set(numeric_cols) | set(categorical_cols)

        excluded = {
            "PatientId",
            "AppointmentID",
            "ScheduledDay",
            "AppointmentDay",
            "Neighbourhood",
            "No-show",
            "target_name",
            "no_show",
        }
        assert all_feature_cols.isdisjoint(excluded)

    def test_categorical_columns_are_the_nominal_ones(self, processed_df):
        _, categorical_cols = get_feature_columns(processed_df)
        assert set(categorical_cols) == {"Gender", "age_group", "neighbourhood_grouped"}


class TestSplitFeaturesTarget:
    def test_returns_x_and_binary_y(self, processed_df):
        X, y = split_features_target(processed_df)
        assert len(X) == len(processed_df)
        assert set(y.unique()).issubset({0, 1})
        assert "no_show" not in X.columns


class TestBuildPreprocessor:
    def test_one_hot_produces_expected_columns(self, processed_df):
        numeric_cols, categorical_cols = get_feature_columns(processed_df)
        X, _ = split_features_target(processed_df)

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(X)

        feature_names = preprocessor.get_feature_names_out()

        # Una columna one-hot por cada categoría de Gender vista en fit.
        assert any(name.startswith("cat__Gender_") for name in feature_names)
        for category in X["Gender"].unique():
            assert f"cat__Gender_{category}" in feature_names

        # Las numéricas se pasan sin transformar (mismo nombre, prefijo num__).
        for col in numeric_cols:
            assert f"num__{col}" in feature_names

    def test_handle_unknown_category_does_not_break_inference(self, processed_df):
        numeric_cols, categorical_cols = get_feature_columns(processed_df)
        X, _ = split_features_target(processed_df)

        train_mask = X["neighbourhood_grouped"] != "GOIABEIRAS"
        X_train = X.loc[train_mask]
        X_test = X.loc[~train_mask].copy()
        # Categoría nueva, jamás vista en fit.
        X_test["neighbourhood_grouped"] = "BAIRRO_NUEVO_NUNCA_VISTO"

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(X_train)

        # No debe lanzar excepción gracias a handle_unknown="ignore".
        transformed = preprocessor.transform(X_test)
        assert transformed.shape[0] == len(X_test)


class TestNoLeakage:
    def test_preprocessor_fits_only_on_train_and_transforms_test_without_refit(
        self, processed_df
    ):
        numeric_cols, categorical_cols = get_feature_columns(processed_df)
        X, _ = split_features_target(processed_df)

        X_train = X.iloc[:60]
        X_test = X.iloc[60:]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(X_train)

        categories_after_fit = [
            list(cats)
            for cats in preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .categories_
        ]

        # Transformar test dos veces no debe alterar las categorías
        # aprendidas ni requerir un nuevo fit.
        first_transform = preprocessor.transform(X_test)
        categories_after_transform = [
            list(cats)
            for cats in preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .categories_
        ]
        second_transform = preprocessor.transform(X_test)

        assert categories_after_fit == categories_after_transform
        np.testing.assert_array_equal(first_transform, second_transform)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


class TestTrainModels:
    def test_holdout_split_is_stratified(self, processed_df):
        result = train_models(processed_df, random_state=42)

        full_ratio = processed_df["no_show"].mean()
        train_ratio = result["y_train"].mean()
        test_ratio = result["y_test"].mean()

        # Con estratificación, la proporción de no_show en train/test debe
        # aproximarse a la del dataset completo.
        assert train_ratio == pytest.approx(full_ratio, abs=0.15)
        assert test_ratio == pytest.approx(full_ratio, abs=0.15)

    def test_pipeline_trains_and_predict_proba_in_unit_interval(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        X_test = result["X_test"]

        proba = pipeline.predict_proba(X_test)[:, 1]

        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_both_models_are_fitted_and_final_model_chosen_by_cv(self, processed_df):
        result = train_models(processed_df, random_state=42)

        assert set(result["pipelines"]) == {"decision_tree", "random_forest"}
        assert result["best_model"] in result["pipelines"]
        assert "f1_mean" in result["cv_results"]["decision_tree"]
        assert "roc_auc_mean" in result["cv_results"]["random_forest"]

    def test_model_and_metrics_are_persisted(self, processed_df, tmp_path):
        result = train_models(processed_df, random_state=42)

        model_path, metrics_path = save_artifacts(result, models_dir=tmp_path)

        assert model_path.exists()
        assert metrics_path.exists()

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["best_model"] == result["best_model"]

        import joblib

        reloaded = joblib.load(model_path)
        proba = reloaded.predict_proba(result["X_test"])[:, 1]
        assert ((proba >= 0.0) & (proba <= 1.0)).all()


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


class TestEvaluateModel:
    def test_classification_report_and_roc_auc_run(self, processed_df):
        result = train_models(processed_df, random_state=42)
        metrics = evaluate_model(result["pipeline"], result["X_test"], result["y_test"])

        assert "accuracy" in metrics["classification_report"]
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["confusion_matrix"].shape == (2, 2)
        assert "fpr" in metrics["roc_curve"]
        assert "tpr" in metrics["roc_curve"]


class TestThresholdAnalysis:
    def test_returns_metrics_per_threshold(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        y_proba = pipeline.predict_proba(result["X_test"])[:, 1]

        thresholds = [0.2, 0.5, 0.8]
        table = threshold_analysis(result["y_test"], y_proba, thresholds=thresholds)

        assert list(table["threshold"]) == thresholds
        assert {"precision", "recall", "f1"}.issubset(table.columns)
        assert ((table["precision"] >= 0.0) & (table["precision"] <= 1.0)).all()
        assert ((table["recall"] >= 0.0) & (table["recall"] <= 1.0)).all()

    def test_lower_threshold_never_decreases_recall(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        y_proba = pipeline.predict_proba(result["X_test"])[:, 1]

        table = threshold_analysis(result["y_test"], y_proba, thresholds=[0.2, 0.8])
        low_threshold_recall = table.loc[table["threshold"] == 0.2, "recall"].iloc[0]
        high_threshold_recall = table.loc[table["threshold"] == 0.8, "recall"].iloc[0]

        assert low_threshold_recall >= high_threshold_recall


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


class TestPredictAppointment:
    def test_returns_float_probability_in_unit_interval(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        features = result["X_test"].iloc[0].to_dict()

        proba = predict_appointment(pipeline, features)

        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0


class TestPredictBatch:
    def test_adds_proba_column_and_ranks_descending(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        X_test = result["X_test"]

        ranked = predict_batch(pipeline, X_test)

        assert "no_show_proba" in ranked.columns
        assert len(ranked) == len(X_test)
        probas = ranked["no_show_proba"].tolist()
        assert probas == sorted(probas, reverse=True)

    def test_does_not_mutate_input_dataframe(self, processed_df):
        result = train_models(processed_df, random_state=42)
        pipeline = result["pipeline"]
        X_test = result["X_test"].copy()

        predict_batch(pipeline, X_test)

        assert "no_show_proba" not in X_test.columns


class TestRecommendAction:
    def test_low_risk_band(self):
        assert recommend_action(config.RISK_LOW - 0.01) == "sin acción"

    def test_medium_risk_band(self):
        midpoint = (config.RISK_LOW + config.RISK_HIGH) / 2
        assert recommend_action(midpoint) == "recordatorio SMS"

    def test_high_risk_band(self):
        assert recommend_action(config.RISK_HIGH) == "llamado + sobreturno"
        assert recommend_action(config.RISK_HIGH + 0.01) == "llamado + sobreturno"

    def test_boundary_at_risk_low_is_medium(self):
        assert recommend_action(config.RISK_LOW) == "recordatorio SMS"
