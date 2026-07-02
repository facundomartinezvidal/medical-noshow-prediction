"""Aplicación funcional interactiva de predicción de no-show de turnos
médicos (TPO Ciencia de Datos, UADE, Grupo 7).

Tres modos:

- **Turno individual**: ingresa los atributos de un turno y muestra la
  probabilidad de no-show y la acción recomendada.
- **Agenda del día (lote)**: sube un CSV con varios turnos y devuelve una
  tabla rankeada por riesgo, con la acción sugerida y una estimación de
  valor de negocio.
- **Dashboard de presentación**: storytelling del EDA, técnica de minería
  (modelo, umbral de decisión interactivo) y conclusión, para la
  exposición de la cátedra (ver `app/dashboard.py`).

Toda la lógica de transformación de datos se delega en `noshow.weather`,
`noshow.features` y `noshow.predict` (el mismo pipeline que usó
`noshow.train`): esta app nunca recodifica manualmente las features. Las
funciones puras que orquestan ese camino viven en `app/logic.py`, donde
están testeadas sin depender de un runtime de Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# La app se ejecuta con `streamlit run app/streamlit_app.py` desde la raíz
# del repo, pero ni la raíz ni `src/` están garantizados en `sys.path` (el
# paquete `noshow` no está instalado como editable). Se agregan ambos de
# forma explícita e idempotente antes de importar nada del proyecto.
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
for _path in (_ROOT_DIR, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import datetime as dt  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import dashboard, logic  # noqa: E402
from noshow import config  # noqa: E402
from noshow.predict import DEFAULT_MODEL_PATH, load_model  # noqa: E402
from noshow.weather import load_weather_daily  # noqa: E402

BATCH_DISPLAY_COLUMNS: list[str] = [
    "Gender",
    "Age",
    "Neighbourhood",
    "AppointmentDay",
    "no_show_proba",
    "accion_recomendada",
]


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def get_model():
    """Carga el pipeline persistido (`models/model.joblib`). Cacheado como
    recurso: se carga una única vez por sesión de la app.
    """
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner="Cargando clima diario cacheado...")
def get_weather_daily() -> pd.DataFrame:
    """Carga el clima diario cacheado (`data/external/weather_daily_a612.csv`)
    vía `noshow.weather.load_weather_daily`. Cacheado como dato: se
    recalcula solo si cambian sus argumentos (nunca, en este caso).
    """
    return load_weather_daily()


def render_model_missing() -> None:
    st.error(
        "No se encontró el modelo entrenado en "
        f"`{DEFAULT_MODEL_PATH.relative_to(config.ROOT_DIR)}`.\n\n"
        "Corré `python -m noshow.train` desde la raíz del proyecto para "
        "entrenarlo y generarlo antes de usar la aplicación."
    )


def render_risk_result(proba: float, action: str) -> None:
    band = logic.risk_band(proba)
    color = logic.RISK_BAND_COLORS[band]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad de no-show", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
    with col2:
        st.markdown(f"**Banda de riesgo:** :{color}[{band.upper()}]")
        if band == "bajo":
            st.success(f"Acción recomendada: {action}")
        elif band == "medio":
            st.warning(f"Acción recomendada: {action}")
        else:
            st.error(f"Acción recomendada: {action}")


def render_single_appointment(model, weather_daily: pd.DataFrame) -> None:
    st.subheader("Datos del turno")

    options = logic.get_categorical_options(model)
    neighbourhood_options = sorted(options.get("neighbourhood_grouped", ["OTHER"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Edad", min_value=0, max_value=115, value=40, step=1)
        gender = st.selectbox("Género", options=sorted(options.get("Gender", ["F", "M"])))
        neighbourhood = st.selectbox(
            "Barrio",
            options=neighbourhood_options,
            help='Elegí "OTHER" si el barrio del turno no figura en la lista '
            "(el modelo agrupa así los barrios de baja frecuencia).",
        )
    with col2:
        today = dt.date.today()
        scheduled_day = st.date_input("Fecha de agendamiento", value=today)
        appointment_day = st.date_input(
            "Fecha del turno", value=today + dt.timedelta(days=7)
        )
        sms_received = st.checkbox("SMS recibido", value=True)
        scholarship = st.checkbox("Beca social (Scholarship)")
    with col3:
        st.markdown("**Comorbilidades**")
        hipertension = st.checkbox("Hipertensión")
        diabetes = st.checkbox("Diabetes")
        alcoholism = st.checkbox("Alcoholismo")
        handcap = st.checkbox("Discapacidad")

    st.subheader("Clima del día del turno")
    auto_weather = logic.lookup_weather_for_date(weather_daily, appointment_day)
    weather_override = None
    if auto_weather is not None:
        wcol1, wcol2, wcol3 = st.columns(3)
        wcol1.metric("Precipitación (mm)", f"{auto_weather['precipitation_mm']:.1f}")
        wcol2.metric("Temp. media (°C)", f"{auto_weather['temp_mean']:.1f}")
        wcol3.metric("¿Llovió?", "Sí" if auto_weather["is_rainy"] else "No")
        st.caption(
            "Clima autocompletado desde el caché diario (estación INMET A612)."
        )
        override_manual = st.checkbox("Sobrescribir clima manualmente")
        if override_manual:
            weather_override = _render_manual_weather_inputs()
    else:
        st.warning(
            "No hay clima cacheado para esa fecha (fuera del rango cubierto "
            "por la estación). Ingresá los valores manualmente (opcional; "
            "si no los completás se usan valores neutros)."
        )
        weather_override = _render_manual_weather_inputs()

    if st.button("Calcular probabilidad de no-show", type="primary"):
        try:
            raw_row = logic.build_appointment_row(
                age=age,
                gender=gender,
                neighbourhood=neighbourhood,
                scheduled_day=scheduled_day,
                appointment_day=appointment_day,
                sms_received=sms_received,
                hipertension=hipertension,
                diabetes=diabetes,
                alcoholism=alcoholism,
                handcap=handcap,
                scholarship=scholarship,
            )
            proba, action = logic.score_single_appointment(
                model, weather_daily, raw_row, weather_override=weather_override
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            render_risk_result(proba, action)


def _render_manual_weather_inputs() -> dict:
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        precipitation_mm = st.number_input(
            "Precipitación (mm)", min_value=0.0, value=0.0, step=0.5
        )
    with mcol2:
        temp_mean = st.number_input(
            "Temperatura media (°C)", min_value=-10.0, max_value=50.0, value=24.0
        )
    with mcol3:
        is_rainy = st.checkbox("¿Día lluvioso?")
    return {
        "precipitation_mm": precipitation_mm,
        "temp_mean": temp_mean,
        "is_rainy": is_rainy,
    }


def render_batch_mode(model, weather_daily: pd.DataFrame) -> None:
    st.subheader("Agenda del día")
    with st.expander("Formato del CSV esperado", expanded=False):
        st.markdown(
            "El archivo debe tener las columnas crudas del dataset original: "
            f"`{'`, `'.join(logic.RAW_APPOINTMENT_COLUMNS)}`."
        )
        st.download_button(
            "Descargar CSV de ejemplo",
            data=logic.sample_batch_csv_bytes(),
            file_name="agenda_ejemplo.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Subí el CSV de la agenda del día", type=["csv"])
    if uploaded is None:
        return

    try:
        raw_df = pd.read_csv(uploaded)
        for date_col in ("ScheduledDay", "AppointmentDay"):
            if date_col in raw_df.columns:
                raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        scored = logic.score_batch(model, weather_daily, raw_df)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - CSV malformado, error de parseo, etc.
        st.error(f"No se pudo procesar el archivo: {exc}")
        return

    n_dropped = len(raw_df) - len(scored)
    if n_dropped > 0:
        st.warning(
            f"{n_dropped} turno(s) del archivo se descartaron por tener "
            "edad o fecha de turno inválidas."
        )

    st.markdown("### Turnos rankeados por riesgo de no-show")
    display_cols = [c for c in BATCH_DISPLAY_COLUMNS if c in scored.columns]
    st.dataframe(
        scored[display_cols].style.format({"no_show_proba": "{:.1%}"}),
        use_container_width=True,
    )

    st.markdown("### Estimación de valor de negocio")
    value = logic.estimate_business_value(scored)
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    vcol1.metric("Turnos de alto riesgo", value["n_alto_riesgo"])
    vcol2.metric(
        "Horas-profesional recuperables", f"{value['horas_profesional_recuperables']:.1f} h"
    )
    vcol3.metric("Valor recuperado estimado", f"AR$ {value['valor_recuperado_ars']:,.0f}")
    vcol4.metric("Sobreturnos sugeridos", value["sobreturnos_sugeridos"])
    st.caption(
        "Supuesto: turno promedio de "
        f"{logic.DEFAULT_APPOINTMENT_DURATION_HOURS * 60:.0f} minutos; "
        f"hora-profesional ociosa valuada en AR$ {config.COST_HOUR:,.0f} "
        "(`noshow.config.COST_HOUR`). Las horas/sobreturnos se calculan "
        "sobre el no-show ESPERADO (suma de probabilidades) entre los "
        "turnos de alto riesgo."
    )

    st.download_button(
        "Descargar agenda scoreada (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="agenda_scoreada.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Predicción de no-show", page_icon="🏥", layout="wide"
    )
    st.title("🏥 Predicción de no-show de turnos médicos")
    st.caption(
        "Herramienta de apoyo para Gerencia Comercial y Técnica: estima la "
        "probabilidad de ausencia de un turno y sugiere la acción preventiva "
        "(recordatorio / llamado + sobreturno controlado)."
    )

    if not DEFAULT_MODEL_PATH.exists():
        render_model_missing()
        return

    model = get_model()
    weather_daily = get_weather_daily()

    modo = st.sidebar.radio(
        "Modo",
        ["Turno individual", "Agenda del día (lote)", "Dashboard de presentación"],
    )
    st.sidebar.caption(
        f"Bandas de riesgo: bajo < {config.RISK_LOW:.0%} · medio "
        f"{config.RISK_LOW:.0%}-{config.RISK_HIGH:.0%} · "
        f"alto ≥ {config.RISK_HIGH:.0%}"
    )

    if modo == "Turno individual":
        render_single_appointment(model, weather_daily)
    elif modo == "Agenda del día (lote)":
        render_batch_mode(model, weather_daily)
    else:
        dashboard.render_dashboard(model)


if __name__ == "__main__":
    main()
