"""Tests que cierran gaps de cobertura contra escenarios GIVEN/WHEN/THEN de
las specs OpenSpec de `noshow-prediction` (`data-pipeline`, `noshow-modeling`):

1. Dataset procesado reproducible (determinismo de `build_features` +
   reuso/escritura del caché de `build_processed_dataset`).
2. `StratifiedKFold`: proporción de clases por fold + agregación
   media/desvío estándar de la utilidad de CV de `train.py`.
3. Poda del Árbol de Decisión: brecha train/test acotada frente a un árbol
   sin podar, y documentación de que el Random Forest no recibe el mismo
   tipo de poda agresiva.
4. Interpretabilidad: smoke test de que las figuras de `noshow.evaluate`
   (matriz de confusión, curva ROC, importancia de variables, árbol) se
   generan como PNG.

Todos los tests usan fixtures sintéticas chicas armadas a mano (o
`sklearn.datasets.make_classification` con `random_state` fijo): NUNCA se
lee el dataset real de 110.527 turnos ni los 4,37M de registros horarios de
clima dentro de un test unitario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from noshow import evaluate, features, train
from noshow.preprocess import build_preprocessor, get_feature_columns, split_features_target
from noshow.train import train_models

# --------------------------------------------------------------------------
# Fixtures compartidas
# --------------------------------------------------------------------------


def _make_raw_appointments_df(n: int = 30, seed: int = 11) -> pd.DataFrame:
    """DataFrame sintético con la forma cruda de turnos (fechas ya
    parseadas a datetime), suficiente para ejercitar `build_features`.
    """
    rng = np.random.default_rng(seed)
    scheduled = pd.date_range("2016-04-20", periods=n, freq="6h")
    lead_time_days = rng.integers(0, 8, size=n)
    appointment = pd.to_datetime(scheduled.normalize()) + pd.to_timedelta(
        lead_time_days, unit="D"
    )
    return pd.DataFrame(
        {
            "PatientId": np.arange(1, n + 1, dtype=float),
            "AppointmentID": np.arange(1000, 1000 + n),
            "Gender": rng.choice(["F", "M"], size=n),
            "ScheduledDay": scheduled,
            "AppointmentDay": appointment,
            "Age": rng.integers(0, 90, size=n),
            "Neighbourhood": rng.choice(
                ["JARDIM DA PENHA", "MATA DA PRAIA", "GOIABEIRAS"], size=n
            ),
            "Scholarship": rng.integers(0, 2, size=n),
            "Hipertension": rng.integers(0, 2, size=n),
            "Diabetes": rng.integers(0, 2, size=n),
            "Alcoholism": rng.integers(0, 2, size=n),
            "Handcap": rng.integers(0, 2, size=n),
            "SMS_received": rng.integers(0, 2, size=n),
        }
    )


def _make_loaded_appointments_df(n: int = 30, seed: int = 7) -> pd.DataFrame:
    """DataFrame con la forma de salida de `noshow.data_load.load_appointments`
    (incluye `No-show`/`no_show`/`target_name` ya normalizados), para
    inyectar como resultado simulado de `load_appointments` sin tocar el
    CSV real.
    """
    df = _make_raw_appointments_df(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    no_show = rng.integers(0, 2, size=n)
    df["No-show"] = np.where(no_show == 1, "Yes", "No")
    df["no_show"] = no_show.astype("int64")
    df["target_name"] = np.where(no_show == 1, "no_show", "show")
    return df


def _make_daily_weather_df(appointment_dates: pd.Series, seed: int = 3) -> pd.DataFrame:
    """DataFrame con la forma de salida de `load_weather_daily` (una fila
    por fecha), cubriendo las fechas de `appointment_dates`.
    """
    unique_dates = pd.to_datetime(
        sorted(pd.to_datetime(appointment_dates).dt.normalize().unique())
    )
    n = len(unique_dates)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": unique_dates,
            "precipitation_mm": rng.uniform(0, 5, size=n),
            "temp_max": rng.uniform(25, 32, size=n),
            "temp_min": rng.uniform(18, 24, size=n),
            "temp_mean": rng.uniform(20, 28, size=n),
            "humidity_mean": rng.uniform(60, 90, size=n),
            "is_rainy": rng.integers(0, 2, size=n),
        }
    )


def _make_processed_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """DataFrame sintético con la forma del dataset procesado real
    (columnas crudas + features de `noshow.features` + clima), con un
    target `no_show` desbalanceado (~20 % positivos, similar al dataset
    real de ~20,2 %). Réplica local del helper equivalente en
    `tests/test_modeling.py` (cada archivo de test mantiene sus propias
    fixtures, siguiendo la convención existente del repo).
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
# Gap 1: Dataset procesado reproducible (spec data-pipeline)
# --------------------------------------------------------------------------


class TestBuildFeaturesDeterministic:
    def test_build_features_is_deterministic(self):
        """`build_features` no tiene aleatoriedad: correrla dos veces sobre
        el mismo input crudo SHALL producir un resultado idéntico
        (Scenario: Dataset procesado reproducible).
        """
        raw = _make_raw_appointments_df()

        first = features.build_features(raw.copy())
        second = features.build_features(raw.copy())

        pd.testing.assert_frame_equal(first, second)


class TestBuildProcessedDatasetCache:
    def test_writes_and_reuses_cache_without_reloading_raw(self, tmp_path, monkeypatch):
        """`build_processed_dataset` SHALL escribir el caché en la primera
        corrida y reusarlo en corridas subsiguientes, sin recargar el
        dataset crudo (Scenario: Dataset procesado reproducible ->
        "reutilizable tanto por el entrenamiento como por la aplicación").
        """
        cache_path = tmp_path / "appointments_processed.csv"
        loaded_appointments = _make_loaded_appointments_df()
        daily_weather = _make_daily_weather_df(loaded_appointments["AppointmentDay"])

        monkeypatch.setattr(
            features, "load_appointments", lambda *a, **k: loaded_appointments.copy()
        )
        monkeypatch.setattr(
            features, "load_weather_daily", lambda *a, **k: daily_weather.copy()
        )

        assert not cache_path.exists()
        first = features.build_processed_dataset(cache=cache_path, use_cache=True)
        assert cache_path.exists(), "la primera corrida (cache-miss) debe escribir el caché"

        # Segunda corrida: si el caché no se reusara, estos loaders
        # explotarían y el test fallaría con un mensaje claro.
        def _boom(*_args, **_kwargs):
            raise AssertionError(
                "build_processed_dataset no debería recargar el dataset crudo "
                "habiendo un caché válido"
            )

        monkeypatch.setattr(features, "load_appointments", _boom)
        monkeypatch.setattr(features, "load_weather_daily", _boom)

        second = features.build_processed_dataset(cache=cache_path, use_cache=True)

        # El round-trip por CSV puede no preservar dtypes no esenciales
        # (categorías, etc.), así que comparamos las columnas clave del
        # dataset procesado en vez de un assert_frame_equal estricto.
        key_cols = [
            "lead_time_days",
            "appointment_dow",
            "appointment_month",
            "age_group",
            "has_comorbidity",
            "neighbourhood_grouped",
            "no_show",
        ]
        assert len(second) == len(first)
        for col in key_cols:
            assert list(second[col]) == list(first[col]), f"columna {col} difiere tras cache-hit"


# --------------------------------------------------------------------------
# Gap 2: StratifiedKFold — proporción por fold + media±std (spec noshow-modeling)
# --------------------------------------------------------------------------


class TestStratifiedKFoldFoldProportions:
    def test_each_fold_preserves_class_ratio(self):
        """Scenario: Proporción de clases por fold — cada fold de test
        SHALL conservar (aproximadamente) el ~80/20 de la clase `no_show`
        del conjunto completo.
        """
        n = 80
        n_pos = round(n * 0.2)
        y = pd.Series([1] * n_pos + [0] * (n - n_pos))
        X = pd.DataFrame({"x": np.arange(n)})
        overall_ratio = y.mean()
        assert overall_ratio == pytest.approx(0.2, abs=0.01)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_ratios = []
        for _, test_idx in cv.split(X, y):
            fold_ratios.append(y.iloc[test_idx].mean())

        assert len(fold_ratios) == 5
        for fold_ratio in fold_ratios:
            # Con 16 filas por fold y 20% de positivos (~3.2), la
            # proporción exacta no es alcanzable: se tolera +/-0.08.
            assert fold_ratio == pytest.approx(overall_ratio, abs=0.08)

    def test_unbalanced_60_40_rows_all_folds_preserve_ratio(self):
        """Variante con tamaño de dataset chico (60 filas) para cubrir el
        rango 60-100 filas pedido: la proporción por fold sigue
        conservándose aproximadamente.
        """
        n = 60
        n_pos = round(n * 0.2)
        rng = np.random.default_rng(0)
        y = pd.Series(rng.permutation([1] * n_pos + [0] * (n - n_pos)))
        X = pd.DataFrame({"x": np.arange(n)})
        overall_ratio = y.mean()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for _, test_idx in cv.split(X, y):
            fold_ratio = y.iloc[test_idx].mean()
            assert fold_ratio == pytest.approx(overall_ratio, abs=0.1)


class TestCrossValidateModelAggregates:
    def test_returns_mean_and_std_for_f1_and_roc_auc(self, processed_df):
        """Scenario: Métricas agregadas entre folds — la utilidad de CV
        real de `train.py` (`_cross_validate_model`) SHALL reportar media
        Y desvío estándar de F1/ROC-AUC entre folds, no un único corte.
        """
        numeric_cols, categorical_cols = get_feature_columns(processed_df)
        X, y = split_features_target(processed_df)

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = train._build_pipeline(
            preprocessor, DecisionTreeClassifier(**train.DT_PARAMS)
        )

        cv_result = train._cross_validate_model(pipeline, X, y, random_state=42)

        assert set(cv_result) == {"f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std"}
        for key, value in cv_result.items():
            assert isinstance(value, float), f"{key} debe ser float, no {type(value)}"
        assert 0.0 <= cv_result["f1_mean"] <= 1.0
        assert 0.0 <= cv_result["roc_auc_mean"] <= 1.0
        assert cv_result["f1_std"] >= 0.0
        assert cv_result["roc_auc_std"] >= 0.0

    def test_train_models_cv_results_expose_mean_and_std_per_model(self, processed_df):
        """Extensión de integración: `train_models` (que usa
        `_cross_validate_model` internamente) expone media±std para AMBOS
        modelos comparados por CV.
        """
        result = train_models(processed_df, random_state=42)

        for model_name in ("decision_tree", "random_forest"):
            cv = result["cv_results"][model_name]
            assert {"f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std"}.issubset(cv)


# --------------------------------------------------------------------------
# Gap 3: Poda del árbol — overfitting train vs test (spec noshow-modeling)
# --------------------------------------------------------------------------


class TestDecisionTreePruning:
    @staticmethod
    def _make_noisy_classification_df(n: int = 600, seed: int = 42):
        X, y = make_classification(
            n_samples=n,
            n_features=10,
            n_informative=4,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.8, 0.2],
            flip_y=0.15,
            random_state=seed,
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return X_df, pd.Series(y)

    def test_pruned_tree_generalizes_better_than_unpruned(self):
        """Scenario: Poda del árbol — el DT podado (hiperparámetros reales
        de `train.DT_PARAMS`) SHALL mostrar una brecha train/test acotada
        (controla el overfitting); un DT sin podar sobreajusta más.
        """
        X, y = self._make_noisy_classification_df()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        pruned = DecisionTreeClassifier(**train.DT_PARAMS)
        pruned.fit(X_train, y_train)
        pruned_train_acc = accuracy_score(y_train, pruned.predict(X_train))
        pruned_test_acc = accuracy_score(y_test, pruned.predict(X_test))
        pruned_gap = pruned_train_acc - pruned_test_acc

        # Árbol sin podar: sin límite de profundidad ni de hojas mínimas,
        # crece hasta memorizar el train set.
        unpruned = DecisionTreeClassifier(random_state=train.RANDOM_STATE)
        unpruned.fit(X_train, y_train)
        unpruned_train_acc = accuracy_score(y_train, unpruned.predict(X_train))
        unpruned_test_acc = accuracy_score(y_test, unpruned.predict(X_test))
        unpruned_gap = unpruned_train_acc - unpruned_test_acc

        # El árbol sin podar memoriza el train set (accuracy casi perfecta).
        assert unpruned_train_acc > 0.95
        # El árbol podado generaliza mejor: brecha train-test acotada.
        assert pruned_gap < 0.15
        # El árbol sin podar sobreajusta más marcadamente que el podado.
        assert unpruned_gap > pruned_gap

    def test_random_forest_params_are_not_pruned_like_the_tree(self):
        """Documenta la diferencia de poda entre modelos: el RF admite
        árboles individuales más profundos y hojas más chicas que el DT
        podado (el control de overfitting lo hace el ensamble, no la poda
        por árbol), conforme al Scenario: "el Random Forest NO se poda".
        """
        assert train.RF_PARAMS["max_depth"] > train.DT_PARAMS["max_depth"]
        assert train.RF_PARAMS["min_samples_leaf"] < train.DT_PARAMS["min_samples_leaf"]
        # El ensamble (múltiples árboles) es el mecanismo real de control
        # de varianza del RF, no una poda agresiva por árbol individual.
        assert train.RF_PARAMS["n_estimators"] > 1


# --------------------------------------------------------------------------
# Gap 4: Interpretabilidad — smoke test de figuras (spec noshow-modeling)
# --------------------------------------------------------------------------


class TestInterpretabilityPlotsSmoke:
    def test_all_plots_are_created_as_png_files(self, processed_df, tmp_path):
        """Scenario: Importancia de variables + Reporte de métricas — las
        figuras de interpretabilidad y evaluación (matriz de confusión,
        curva ROC, importancia de variables del RF, árbol podado) SHALL
        generarse como archivos PNG, usando el backend no interactivo
        `Agg` (fijado por `noshow.evaluate` al importarse).
        """
        result = train_models(processed_df, random_state=42)
        rf_pipeline = result["pipelines"]["random_forest"]
        dt_pipeline = result["pipelines"]["decision_tree"]

        metrics = evaluate.evaluate_model(rf_pipeline, result["X_test"], result["y_test"])

        cm_path = evaluate.plot_confusion_matrix(
            metrics["confusion_matrix"], output_path=tmp_path / "confusion_matrix.png"
        )
        roc_path = evaluate.plot_roc_curve(
            metrics["roc_curve"]["fpr"],
            metrics["roc_curve"]["tpr"],
            metrics["roc_auc"],
            output_path=tmp_path / "roc_curve.png",
        )
        fi_path = evaluate.plot_feature_importances(
            rf_pipeline, output_path=tmp_path / "feature_importances.png"
        )
        tree_path = evaluate.plot_decision_tree(
            dt_pipeline, output_path=tmp_path / "decision_tree.png"
        )

        for path in (cm_path, roc_path, fi_path, tree_path):
            assert path.exists(), f"{path} no fue generado"
            assert path.stat().st_size > 0, f"{path} está vacío"

    def test_plot_feature_importances_rejects_model_without_importances(self, tmp_path):
        """Caso negativo: `plot_feature_importances` requiere un
        clasificador con `feature_importances_` (ej. un
        `RandomForestClassifier`); con un objeto que no lo expone, SHALL
        fallar con un `ValueError` explícito en vez de romper silenciosamente.
        """

        class _FakePipeline:
            named_steps = {"classifier": object()}

        with pytest.raises(ValueError):
            evaluate.plot_feature_importances(
                _FakePipeline(), output_path=tmp_path / "feature_importances.png"
            )
