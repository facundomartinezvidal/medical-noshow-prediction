"""Tests de la capa de datos (`noshow.data_load`, `noshow.weather`,
`noshow.features`).

Todos los tests usan fixtures sintéticas pequeñas (DataFrames armados a mano
o CSVs temporales de pocas filas): NUNCA se leen los 110.527 turnos ni los
4,37M de registros horarios de clima reales dentro de un test unitario.
"""

from __future__ import annotations

import pandas as pd
import pytest

from noshow import config
from noshow.data_load import load_appointments, validate_schema
from noshow.features import (
    add_age_group,
    add_comorbidity_features,
    add_date_features,
    build_features,
    clean_invalid_records,
    compute_lead_time_days,
    group_rare_neighbourhoods,
)
from noshow.weather import load_weather_daily, merge_weather


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _make_appointments_df(**overrides) -> pd.DataFrame:
    """Construye un DataFrame sintético de turnos con las 14 columnas
    reales del dataset. `overrides` permite pisar columnas puntuales con
    listas de igual longitud.
    """
    base = {
        "PatientId": [1.0, 2.0, 3.0, 4.0, 5.0],
        "AppointmentID": [100, 101, 102, 103, 104],
        "Gender": ["F", "M", "F", "M", "F"],
        "ScheduledDay": [
            "2016-04-27T08:36:51Z",
            "2016-04-27T15:05:12Z",
            "2016-04-29T08:00:00Z",
            "2016-04-29T09:00:00Z",
            "2016-04-30T10:00:00Z",
        ],
        "AppointmentDay": [
            "2016-04-29T00:00:00Z",
            "2016-04-29T00:00:00Z",
            "2016-04-29T00:00:00Z",
            "2016-04-29T00:00:00Z",
            "2016-05-02T00:00:00Z",
        ],
        "Age": [62, 23, 8, 76, 39],
        "Neighbourhood": [
            "JARDIM DA PENHA",
            "JARDIM DA PENHA",
            "MATA DA PRAIA",
            "JARDIM DA PENHA",
            "GOIABEIRAS",
        ],
        "Scholarship": [0, 0, 0, 0, 1],
        "Hipertension": [1, 0, 0, 1, 0],
        "Diabetes": [0, 0, 0, 1, 0],
        "Alcoholism": [0, 0, 0, 0, 0],
        "Handcap": [0, 0, 0, 0, 0],
        "SMS_received": [0, 1, 0, 0, 1],
        "No-show": ["No", "Yes", "No", "No", "Yes"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_hourly_weather_df() -> pd.DataFrame:
    """Fixture de ~48 filas horarias (2 días x 24 horas) para la estación
    A612, más un puñado de filas de otra estación para probar el filtrado.
    """
    dates = ["2016-04-29"] * 24 + ["2016-04-30"] * 24
    hours = [f"{h:02d}:00" for h in range(24)] * 2

    # Día 1 (2016-04-29): sin lluvia. Día 2 (2016-04-30): con lluvia.
    precip_day1 = [0.0] * 24
    precip_day2 = [0.0] * 23 + [2.5]  # una hora con lluvia -> is_rainy=1
    precip = precip_day1 + precip_day2

    temp_mean = [20.0 + (h % 5) for h in range(24)] * 2
    temp_max = [t + 2 for t in temp_mean]
    temp_min = [t - 2 for t in temp_mean]
    humidity = [70.0 + (h % 10) for h in range(24)] * 2

    a612 = pd.DataFrame(
        {
            "DATA (YYYY-MM-DD)": dates,
            "Hora UTC": hours,
            "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": precip,
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": temp_mean,
            "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": temp_max,
            "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": temp_min,
            "UMIDADE RELATIVA DO AR, HORARIA (%)": humidity,
            "ESTACAO": ["A612"] * 48,
        }
    )

    other_station = pd.DataFrame(
        {
            "DATA (YYYY-MM-DD)": ["2016-04-29", "2016-04-30"],
            "Hora UTC": ["00:00", "00:00"],
            "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": [10.0, 10.0],
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": [15.0, 15.0],
            "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": [16.0, 16.0],
            "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": [14.0, 14.0],
            "UMIDADE RELATIVA DO AR, HORARIA (%)": [50.0, 50.0],
            "ESTACAO": ["A001", "A001"],
        }
    )

    return pd.concat([a612, other_station], ignore_index=True)


# --------------------------------------------------------------------------
# data_load
# --------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema_passes(self):
        df = _make_appointments_df()
        validate_schema(df)  # no debe lanzar

    def test_missing_column_raises_value_error(self):
        df = _make_appointments_df().drop(columns=["Age", "SMS_received"])
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df)
        message = str(exc_info.value)
        assert "Age" in message
        assert "SMS_received" in message


class TestLoadAppointments:
    def test_normalizes_target_yes_no_to_1_0(self, tmp_path):
        df = _make_appointments_df()
        csv_path = tmp_path / "appointments.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_appointments(csv_path)

        assert loaded["no_show"].tolist() == [0, 1, 0, 0, 1]
        assert loaded["target_name"].tolist() == [
            "show",
            "no_show",
            "show",
            "show",
            "no_show",
        ]

    def test_parses_date_columns_as_datetime(self, tmp_path):
        df = _make_appointments_df()
        csv_path = tmp_path / "appointments.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_appointments(csv_path)

        assert pd.api.types.is_datetime64_any_dtype(loaded["ScheduledDay"])
        assert pd.api.types.is_datetime64_any_dtype(loaded["AppointmentDay"])

    def test_missing_schema_column_raises_before_loading(self, tmp_path):
        df = _make_appointments_df().drop(columns=["Neighbourhood"])
        csv_path = tmp_path / "appointments.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Neighbourhood"):
            load_appointments(csv_path)


# --------------------------------------------------------------------------
# features: limpieza y derivación
# --------------------------------------------------------------------------


class TestComputeLeadTimeDays:
    def test_lead_time_days_calculation(self):
        df = pd.DataFrame(
            {
                "ScheduledDay": pd.to_datetime(
                    ["2016-04-27T08:36:51Z", "2016-04-29T00:00:00Z"]
                ),
                "AppointmentDay": pd.to_datetime(
                    ["2016-04-29T00:00:00Z", "2016-04-29T00:00:00Z"]
                ),
            }
        )
        result = compute_lead_time_days(df)
        assert result["lead_time_days"].tolist() == [2, 0]


class TestCleanInvalidRecords:
    def test_negative_age_is_dropped_and_counted(self, caplog):
        df = pd.DataFrame(
            {
                "Age": [10, -1, 30],
                "lead_time_days": [1, 2, 3],
            }
        )
        with caplog.at_level("INFO"):
            cleaned = clean_invalid_records(df)

        assert len(cleaned) == 2
        assert (cleaned["Age"] >= 0).all()
        assert "1" in caplog.text  # conteo logueado

    def test_negative_lead_time_is_dropped_and_counted(self, caplog):
        df = pd.DataFrame(
            {
                "Age": [10, 20, 30],
                "lead_time_days": [1, -3, 5],
            }
        )
        with caplog.at_level("INFO"):
            cleaned = clean_invalid_records(df)

        assert len(cleaned) == 2
        assert (cleaned["lead_time_days"] >= 0).all()

    def test_no_invalid_records_keeps_all_rows(self):
        df = pd.DataFrame({"Age": [10, 20], "lead_time_days": [1, 2]})
        cleaned = clean_invalid_records(df)
        assert len(cleaned) == 2


class TestAddAgeGroup:
    @pytest.mark.parametrize(
        "age,expected_group",
        [
            (0, "menor"),
            (17, "menor"),
            (18, "adulto"),
            (64, "adulto"),
            (65, "adulto_mayor"),
            (99, "adulto_mayor"),
        ],
    )
    def test_age_binning(self, age, expected_group):
        df = pd.DataFrame({"Age": [age]})
        result = add_age_group(df)
        assert result["age_group"].iloc[0] == expected_group


class TestAddDateFeatures:
    def test_dow_month_and_same_day(self):
        df = pd.DataFrame(
            {
                "AppointmentDay": pd.to_datetime(["2016-04-29T00:00:00Z"]),
                "lead_time_days": [0],
            }
        )
        result = add_date_features(df)
        # 2016-04-29 es viernes -> dayofweek == 4
        assert result["appointment_dow"].iloc[0] == 4
        assert result["appointment_month"].iloc[0] == 4
        assert result["same_day"].iloc[0] == 1

    def test_same_day_is_zero_when_lead_time_positive(self):
        df = pd.DataFrame(
            {
                "AppointmentDay": pd.to_datetime(["2016-04-29T00:00:00Z"]),
                "lead_time_days": [3],
            }
        )
        result = add_date_features(df)
        assert result["same_day"].iloc[0] == 0


class TestAddComorbidityFeatures:
    def test_flags_comorbidity_when_any_condition_present(self):
        df = pd.DataFrame(
            {
                "Hipertension": [1, 0, 0],
                "Diabetes": [0, 1, 0],
                "Alcoholism": [0, 0, 0],
                "Handcap": [0, 0, 2],
            }
        )
        result = add_comorbidity_features(df)
        assert result["has_comorbidity"].tolist() == [1, 1, 1]
        assert result["comorbidity_count"].tolist() == [1, 1, 1]

    def test_no_comorbidity(self):
        df = pd.DataFrame(
            {
                "Hipertension": [0],
                "Diabetes": [0],
                "Alcoholism": [0],
                "Handcap": [0],
            }
        )
        result = add_comorbidity_features(df)
        assert result["has_comorbidity"].iloc[0] == 0
        assert result["comorbidity_count"].iloc[0] == 0


class TestGroupRareNeighbourhoods:
    def test_rare_neighbourhoods_grouped_as_other(self):
        # 9 filas de "COMMON" (90%) y 1 fila de "RARE" (10%) con umbral 15%
        df = pd.DataFrame(
            {"Neighbourhood": ["COMMON"] * 9 + ["RARE"]}
        )
        result = group_rare_neighbourhoods(df, min_freq=0.15)
        assert (result.loc[result["Neighbourhood"] == "COMMON", "neighbourhood_grouped"] == "COMMON").all()
        assert result.loc[result["Neighbourhood"] == "RARE", "neighbourhood_grouped"].iloc[0] == "OTHER"

    def test_frequent_neighbourhoods_kept_as_is(self):
        df = pd.DataFrame({"Neighbourhood": ["A", "A", "B", "B"]})
        result = group_rare_neighbourhoods(df, min_freq=0.1)
        assert set(result["neighbourhood_grouped"]) == {"A", "B"}


class TestBuildFeaturesIntegration:
    def test_build_features_end_to_end_on_small_frame(self):
        df = _make_appointments_df()
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

        result = build_features(df)

        expected_cols = {
            "lead_time_days",
            "appointment_dow",
            "appointment_month",
            "same_day",
            "age_group",
            "comorbidity_count",
            "has_comorbidity",
            "neighbourhood_grouped",
        }
        assert expected_cols.issubset(result.columns)
        assert len(result) == len(df)  # sin registros inválidos en la fixture


# --------------------------------------------------------------------------
# weather: agregación horario -> diario y merge
# --------------------------------------------------------------------------


class TestLoadWeatherDaily:
    def test_reads_by_chunks_filters_station_and_aggregates(self, tmp_path):
        hourly_df = _make_hourly_weather_df()
        raw_csv = tmp_path / "weather-dataset.csv"
        hourly_df.to_csv(raw_csv, index=False)
        cache_csv = tmp_path / "weather_daily_a612.csv"

        # chunksize chico (< filas totales) para forzar múltiples chunks
        daily = load_weather_daily(
            path=raw_csv,
            station="A612",
            cache=cache_csv,
            chunksize=10,
            use_cache=False,
        )

        assert len(daily) == 2  # 2 días
        assert set(daily.columns) == {
            "date",
            "precipitation_mm",
            "temp_max",
            "temp_min",
            "temp_mean",
            "humidity_mean",
            "is_rainy",
        }
        # sólo A612 debe estar representada (precip != 10.0 de la otra estación)
        assert daily["precipitation_mm"].max() < 10.0

    def test_is_rainy_flag(self, tmp_path):
        hourly_df = _make_hourly_weather_df()
        raw_csv = tmp_path / "weather-dataset.csv"
        hourly_df.to_csv(raw_csv, index=False)
        cache_csv = tmp_path / "weather_daily_a612.csv"

        daily = load_weather_daily(
            path=raw_csv, station="A612", cache=cache_csv, chunksize=15, use_cache=False
        )
        daily = daily.sort_values("date").reset_index(drop=True)

        assert daily.loc[0, "is_rainy"] == 0  # 2016-04-29 sin lluvia
        assert daily.loc[1, "is_rainy"] == 1  # 2016-04-30 con lluvia
        assert daily.loc[1, "precipitation_mm"] == pytest.approx(2.5)

    def test_uses_cache_when_present(self, tmp_path):
        cache_csv = tmp_path / "weather_daily_a612.csv"
        cached_daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2016-04-29"]),
                "precipitation_mm": [0.0],
                "temp_max": [25.0],
                "temp_min": [20.0],
                "temp_mean": [22.0],
                "humidity_mean": [70.0],
                "is_rainy": [0],
            }
        )
        cached_daily.to_csv(cache_csv, index=False)

        # path apunta a un CSV inexistente: si no usara el caché, fallaría
        result = load_weather_daily(
            path=tmp_path / "does-not-exist.csv",
            station="A612",
            cache=cache_csv,
            use_cache=True,
        )
        assert len(result) == 1


class TestMergeWeather:
    def test_merge_by_appointment_date(self):
        appointments = pd.DataFrame(
            {
                "AppointmentDay": pd.to_datetime(
                    ["2016-04-29T00:00:00Z", "2016-04-30T00:00:00Z"]
                ),
            }
        )
        weather_daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2016-04-29", "2016-04-30"]),
                "precipitation_mm": [0.0, 5.0],
                "temp_max": [30.0, 28.0],
                "temp_min": [22.0, 21.0],
                "temp_mean": [26.0, 24.0],
                "humidity_mean": [70.0, 85.0],
                "is_rainy": [0, 1],
            }
        )
        merged = merge_weather(appointments, weather_daily)

        assert merged.loc[0, "is_rainy"] == 0
        assert merged.loc[1, "is_rainy"] == 1
        assert merged.loc[1, "precipitation_mm"] == pytest.approx(5.0)
        assert merged["weather_missing"].tolist() == [0, 0]

    def test_missing_weather_date_is_handled_without_breaking(self):
        appointments = pd.DataFrame(
            {
                "AppointmentDay": pd.to_datetime(
                    ["2016-04-29T00:00:00Z", "2099-01-01T00:00:00Z"]
                ),
            }
        )
        weather_daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2016-04-29"]),
                "precipitation_mm": [1.0],
                "temp_max": [30.0],
                "temp_min": [22.0],
                "temp_mean": [26.0],
                "humidity_mean": [70.0],
                "is_rainy": [1],
            }
        )

        merged = merge_weather(appointments, weather_daily)

        assert merged["weather_missing"].tolist() == [0, 1]
        # el faltante no debe romper el pipeline ni dejar NaN sin manejar
        assert not merged.loc[1, ["precipitation_mm", "is_rainy"]].isna().any()
        assert merged.loc[1, "is_rainy"] == 0  # imputado a "no lluvia"
