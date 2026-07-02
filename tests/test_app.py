"""Tests de la aplicación interactiva (`app.streamlit_app`, `app.logic`).

Streamlit no se testea fácil a nivel de UI: se testean las funciones
puras extraídas a `app/logic.py` (armado de la fila cruda de un turno,
scoring individual/lote y estimación de valor de negocio), más un test de
importación que verifica que `app.streamlit_app` no ejecuta la UI al ser
importado (el cuerpo de la app está protegido por `if __name__ ==
"__main__"`).

Como en `tests/test_modeling.py`, se usa un DataFrame sintético pequeño
con la forma del dataset crudo (o del dataset procesado, según el test):
NUNCA se lee el dataset real de 110.521 filas dentro de un test unitario.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app import logic
from noshow import config
from noshow.predict import predict_batch
from noshow.preprocess import build_preprocessor, get_feature_columns, split_features_target
from noshow.train import train_models


# --------------------------------------------------------------------------
# Fixtures: dataset procesado sintético + pipeline entrenado sobre él
# --------------------------------------------------------------------------


def _make_processed_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Mismo generador sintético que `tests/test_modeling.py`: dataset
    procesado (columnas crudas + features + clima) con target
    desbalanceado ~20% positivos.
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


@pytest.fixture()
def trained_pipeline(processed_df):
    result = train_models(processed_df, random_state=42)
    return result["pipeline"]


@pytest.fixture()
def weather_daily_df() -> pd.DataFrame:
    """Clima diario sintético que cubre una ventana de fechas conocida,
    con la misma forma que `noshow.weather.load_weather_daily`.
    """
    dates = pd.date_range("2016-04-20", periods=60, freq="D")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "date": dates,
            "precipitation_mm": rng.uniform(0, 5, size=len(dates)),
            "temp_max": rng.uniform(25, 32, size=len(dates)),
            "temp_min": rng.uniform(18, 24, size=len(dates)),
            "temp_mean": rng.uniform(20, 28, size=len(dates)),
            "humidity_mean": rng.uniform(60, 90, size=len(dates)),
            "is_rainy": rng.integers(0, 2, size=len(dates)),
        }
    )


# --------------------------------------------------------------------------
# Importación del módulo de la app (no debe ejecutar la UI)
# --------------------------------------------------------------------------


class TestAppImport:
    def test_streamlit_app_module_imports_without_running_ui(self):
        """Importar `app.streamlit_app` no debe lanzar excepciones ni
        intentar levantar un servidor Streamlit: el cuerpo de la UI está
        protegido por `if __name__ == "__main__"`.
        """
        import app.streamlit_app as streamlit_app

        assert hasattr(streamlit_app, "main")
        assert callable(streamlit_app.main)


# --------------------------------------------------------------------------
# build_appointment_row
# --------------------------------------------------------------------------


class TestBuildAppointmentRow:
    def test_returns_dataframe_with_expected_raw_columns(self):
        row = logic.build_appointment_row(
            age=40,
            gender="F",
            neighbourhood="JARDIM DA PENHA",
            scheduled_day=dt.date(2016, 4, 20),
            appointment_day=dt.date(2016, 4, 27),
            sms_received=True,
            hipertension=False,
            diabetes=False,
            alcoholism=False,
            handcap=False,
            scholarship=False,
        )

        assert list(row.columns) == logic.RAW_APPOINTMENT_COLUMNS
        assert len(row) == 1
        assert row.loc[0, "Age"] == 40
        assert row.loc[0, "Gender"] == "F"
        assert row.loc[0, "Neighbourhood"] == "JARDIM DA PENHA"
        assert row.loc[0, "SMS_received"] == 1

    def test_rejects_negative_age(self):
        with pytest.raises(ValueError):
            logic.build_appointment_row(
                age=-1,
                gender="F",
                neighbourhood="OTHER",
                scheduled_day=dt.date(2016, 4, 20),
                appointment_day=dt.date(2016, 4, 27),
                sms_received=False,
                hipertension=False,
                diabetes=False,
                alcoholism=False,
                handcap=False,
                scholarship=False,
            )

    def test_rejects_appointment_before_scheduled(self):
        with pytest.raises(ValueError):
            logic.build_appointment_row(
                age=40,
                gender="F",
                neighbourhood="OTHER",
                scheduled_day=dt.date(2016, 5, 1),
                appointment_day=dt.date(2016, 4, 27),
                sms_received=False,
                hipertension=False,
                diabetes=False,
                alcoholism=False,
                handcap=False,
                scholarship=False,
            )


# --------------------------------------------------------------------------
# score_single_appointment (reutiliza merge_weather + build_features)
# --------------------------------------------------------------------------


class TestScoreSingleAppointment:
    def test_returns_probability_and_action_consistent_with_recommend_action(
        self, trained_pipeline, weather_daily_df
    ):
        row = logic.build_appointment_row(
            age=70,
            gender="F",
            neighbourhood="JARDIM DA PENHA",
            scheduled_day=dt.date(2016, 4, 20),
            appointment_day=dt.date(2016, 4, 30),
            sms_received=False,
            hipertension=True,
            diabetes=False,
            alcoholism=False,
            handcap=False,
            scholarship=False,
        )

        proba, action = logic.score_single_appointment(
            trained_pipeline, weather_daily_df, row
        )

        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0
        assert action == logic.recommend_action(proba)

    def test_manual_weather_override_is_applied(self, trained_pipeline, weather_daily_df):
        row = logic.build_appointment_row(
            age=40,
            gender="M",
            neighbourhood="OTHER",
            scheduled_day=dt.date(2016, 4, 20),
            appointment_day=dt.date(2016, 4, 25),
            sms_received=True,
            hipertension=False,
            diabetes=False,
            alcoholism=False,
            handcap=False,
            scholarship=False,
        )

        # No debe lanzar excepción y debe devolver una probabilidad válida
        # aun forzando un override climático manual.
        proba, action = logic.score_single_appointment(
            trained_pipeline,
            weather_daily_df,
            row,
            weather_override={
                "precipitation_mm": 12.0,
                "is_rainy": True,
                "temp_mean": 18.0,
            },
        )
        assert 0.0 <= proba <= 1.0
        assert action in {"sin acción", "recordatorio SMS", "llamado + sobreturno"}

    def test_date_outside_weather_cache_still_scores(self, trained_pipeline, weather_daily_df):
        """Un turno con fecha fuera del rango cubierto por el caché de
        clima no debe romper el scoring: `merge_weather` imputa clima
        neutro y marca `weather_missing=1`.
        """
        row = logic.build_appointment_row(
            age=50,
            gender="F",
            neighbourhood="OTHER",
            scheduled_day=dt.date(2020, 1, 1),
            appointment_day=dt.date(2020, 1, 10),
            sms_received=False,
            hipertension=False,
            diabetes=False,
            alcoholism=False,
            handcap=False,
            scholarship=False,
        )
        proba, action = logic.score_single_appointment(
            trained_pipeline, weather_daily_df, row
        )
        assert 0.0 <= proba <= 1.0


class TestLookupWeatherForDate:
    def test_finds_known_date(self, weather_daily_df):
        result = logic.lookup_weather_for_date(weather_daily_df, dt.date(2016, 4, 21))
        assert result is not None
        assert set(result) == {"precipitation_mm", "is_rainy", "temp_mean"}

    def test_returns_none_for_unknown_date(self, weather_daily_df):
        result = logic.lookup_weather_for_date(weather_daily_df, dt.date(2099, 1, 1))
        assert result is None


class TestRiskBand:
    def test_low_band(self):
        assert logic.risk_band(config.RISK_LOW - 0.01) == "bajo"

    def test_medium_band(self):
        midpoint = (config.RISK_LOW + config.RISK_HIGH) / 2
        assert logic.risk_band(midpoint) == "medio"

    def test_high_band(self):
        assert logic.risk_band(config.RISK_HIGH) == "alto"


class TestGetCategoricalOptions:
    def test_returns_gender_and_neighbourhood_categories_seen_in_training(
        self, trained_pipeline, processed_df
    ):
        options = logic.get_categorical_options(trained_pipeline)

        assert "Gender" in options
        assert "neighbourhood_grouped" in options
        assert set(processed_df["Gender"].unique()).issubset(set(options["Gender"]))
        assert set(processed_df["neighbourhood_grouped"].unique()).issubset(
            set(options["neighbourhood_grouped"])
        )


# --------------------------------------------------------------------------
# score_batch / estimate_business_value (modo lote)
# --------------------------------------------------------------------------


class TestScoreBatch:
    def _raw_batch(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Gender": ["F", "M", "F"],
                "ScheduledDay": [
                    dt.date(2016, 4, 20),
                    dt.date(2016, 4, 21),
                    dt.date(2016, 4, 22),
                ],
                "AppointmentDay": [
                    dt.date(2016, 4, 27),
                    dt.date(2016, 4, 25),
                    dt.date(2016, 4, 22),
                ],
                "Age": [65, 30, 10],
                "Neighbourhood": ["JARDIM DA PENHA", "MATA DA PRAIA", "GOIABEIRAS"],
                "Scholarship": [0, 1, 0],
                "Hipertension": [1, 0, 0],
                "Diabetes": [0, 0, 0],
                "Alcoholism": [0, 0, 0],
                "Handcap": [0, 0, 0],
                "SMS_received": [0, 1, 1],
            }
        )

    def test_ranks_by_descending_risk_and_adds_action(
        self, trained_pipeline, weather_daily_df
    ):
        scored = logic.score_batch(trained_pipeline, weather_daily_df, self._raw_batch())

        assert "no_show_proba" in scored.columns
        assert "accion_recomendada" in scored.columns
        assert len(scored) == 3
        probas = scored["no_show_proba"].tolist()
        assert probas == sorted(probas, reverse=True)
        assert set(scored["accion_recomendada"]).issubset(
            {"sin acción", "recordatorio SMS", "llamado + sobreturno"}
        )

    def test_missing_required_column_raises(self, trained_pipeline, weather_daily_df):
        bad_df = self._raw_batch().drop(columns=["Neighbourhood"])
        with pytest.raises(ValueError):
            logic.score_batch(trained_pipeline, weather_daily_df, bad_df)


class TestSampleBatchCsvBytes:
    def test_produces_valid_csv_with_expected_columns(self):
        import io

        content = logic.sample_batch_csv_bytes()
        assert isinstance(content, bytes)

        df = pd.read_csv(io.BytesIO(content))
        assert list(df.columns) == logic.RAW_APPOINTMENT_COLUMNS
        assert len(df) >= 1


class TestEstimateBusinessValue:
    def test_computes_expected_keys_on_small_dataframe(self):
        scored = pd.DataFrame({"no_show_proba": [0.9, 0.7, 0.4, 0.1]})

        value = logic.estimate_business_value(
            scored, risk_low=0.3, risk_high=0.6, cost_hour=100.0, duration_hours=1.0
        )

        assert value["n_total"] == 4
        assert value["n_alto_riesgo"] == 2  # 0.9 y 0.7
        assert value["n_riesgo_medio"] == 1  # 0.4
        assert value["n_bajo_riesgo"] == 1  # 0.1
        # Esperado de no-shows entre los de alto riesgo: 0.9 + 0.7 = 1.6
        assert value["no_shows_esperados_alto_riesgo"] == pytest.approx(1.6)
        assert value["sobreturnos_sugeridos"] == 2  # round(1.6)
        # horas = 1.6 * 1.0 ; valor = 1.6 * 100
        assert value["horas_profesional_recuperables"] == pytest.approx(1.6)
        assert value["valor_recuperado_ars"] == pytest.approx(160.0)

    def test_raises_without_proba_column(self):
        with pytest.raises(ValueError):
            logic.estimate_business_value(pd.DataFrame({"foo": [1, 2, 3]}))

    def test_uses_project_defaults_when_not_overridden(self):
        scored = pd.DataFrame({"no_show_proba": [0.95]})
        value = logic.estimate_business_value(scored)

        expected_hours = 0.95 * logic.DEFAULT_APPOINTMENT_DURATION_HOURS
        assert value["horas_profesional_recuperables"] == pytest.approx(
            round(expected_hours, 2)
        )
        assert value["valor_recuperado_ars"] == pytest.approx(
            round(expected_hours * config.COST_HOUR, 2)
        )
