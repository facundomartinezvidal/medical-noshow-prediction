"""Dashboard de presentación: storytelling + técnica de minería en vivo.

Pensado para la exposición de 15 minutos ante la Gerencia Comercial y
Técnica (consigna de la cátedra): reconstruye, de forma navegable e
interactiva dentro de la misma app, el contenido de
`notebooks/01_eda_modeling.ipynb` (contexto de negocio, EDA con
storytelling, técnica de minería/modelo y conclusión) sin necesidad de
alternar con el notebook o una presentación estática durante la demo.

Reutiliza el dataset procesado cacheado y el pipeline ya entrenado (mismos
artefactos que usa el resto de la app); no reentrena ni recodifica
features.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from noshow import config
from noshow.evaluate import threshold_analysis
from noshow.features import build_processed_dataset
from noshow.preprocess import split_features_target
from noshow.train import RANDOM_STATE, TEST_SIZE

def _light_theme() -> dict:
    """Tema Altair con fondo blanco y texto oscuro fijos, independiente del
    tema (claro/oscuro) que tenga configurado Streamlit — los gráficos deben
    verse igual de legibles en la presentación sin importar esa opción.
    """
    text_color = "#1f2430"
    grid_color = "#e3e6ea"
    return {
        "config": {
            "background": "white",
            "view": {"stroke": "transparent", "fill": "white"},
            "title": {"color": text_color, "fontSize": 14},
            "axis": {
                "labelColor": text_color,
                "titleColor": text_color,
                "gridColor": grid_color,
                "domainColor": "#9aa1ac",
                "tickColor": "#9aa1ac",
            },
            "legend": {"labelColor": text_color, "titleColor": text_color},
            "header": {"labelColor": text_color, "titleColor": text_color},
        }
    }


alt.themes.register("noshow_light", _light_theme)
alt.themes.enable("noshow_light")

AGE_GROUP_ORDER: list[str] = ["menor", "adulto", "adulto_mayor"]
DOW_LABELS: list[str] = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]


@st.cache_data(show_spinner="Cargando dataset procesado (turnos + clima)...")
def load_processed_dataset() -> pd.DataFrame:
    return build_processed_dataset(use_cache=True)


@st.cache_data(show_spinner="Reproduciendo el hold-out de test del entrenamiento...")
def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce EXACTAMENTE la partición hold-out de `noshow.train.train_models`
    (mismo `test_size`/`random_state`/`stratify`), para poder evaluar el
    pipeline ya persistido sin volver a entrenar nada.
    """
    df = load_processed_dataset()
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


@st.cache_data(show_spinner="Scoreando el hold-out de test...")
def compute_test_proba(_model) -> np.ndarray:
    X_test, _ = load_test_split()
    return _model.predict_proba(X_test)[:, 1]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    metrics_path = config.MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_feature_importances(_model, top_n: int = 15) -> pd.DataFrame:
    classifier = _model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = _model.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [n.split("__", 1)[-1] for n in names]
    out = pd.DataFrame(
        {"feature": clean_names, "importance": classifier.feature_importances_}
    )
    return out.sort_values("importance", ascending=False).head(top_n)


# --- Tab 1: contexto -------------------------------------------------------


def render_contexto(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("El problema, en números")
    n_total = len(df)
    n_noshow = int(df["no_show"].sum())
    pct_noshow = n_noshow / n_total * 100
    best_model = metrics.get("best_model", "random_forest")
    roc_auc = metrics.get("test_metrics", {}).get(best_model, {}).get("roc_auc")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Turnos analizados", f"{n_total:,}")
    col2.metric("Tasa de no-show", f"{pct_noshow:.1f}%", help="22.319 turnos ausentes sobre 110.527 agendados")
    col3.metric("ROC-AUC (modelo elegido)", f"{roc_auc:.3f}" if roc_auc else "—")
    col4.metric("Fuentes de datos", "2", help="Turnos médicos (Kaggle) + clima diario (INMET, estación A612)")

    st.markdown(
        """
Cada no-show implica un **doble costo**: una franja del profesional queda
ociosa y otro paciente en lista de espera pierde la oportunidad de
ocuparla. Hoy la institución no cuenta con información anticipada para
actuar de forma preventiva.

**Propuesta de valor:** estimar, *al momento de agendar el turno*, la
probabilidad de no-show para habilitar dos acciones dirigidas a los
turnos de mayor riesgo — recordatorio reforzado y sobreturno controlado —
en lugar de aplicarlas de forma uniforme a toda la agenda.
        """
    )


# --- Tab 2: EDA / storytelling ----------------------------------------------


def render_eda(df: pd.DataFrame) -> None:
    st.subheader("¿Qué distingue a quien falta de quien asiste?")

    target_counts = (
        df["target_name"].value_counts(normalize=True).mul(100).rename("pct").reset_index()
    )
    target_counts.columns = ["target", "pct"]
    chart = (
        alt.Chart(target_counts)
        .mark_bar()
        .encode(
            x=alt.X("target:N", title="Target", sort=["show", "no_show"]),
            y=alt.Y("pct:Q", title="% de turnos"),
            color=alt.Color("target:N", legend=None),
            tooltip=[alt.Tooltip("target:N"), alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(height=280, title="Distribución del target (desbalance ~80/20)")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "El desbalance obliga a evaluar con precision/recall/ROC-AUC, no con "
        "accuracy: un modelo trivial que siempre prediga 'show' acertaría ~80% "
        "sin aportar valor de negocio."
    )

    col1, col2 = st.columns(2)
    with col1:
        lead_time_clip = df[df["lead_time_days"] <= 120]
        box = (
            alt.Chart(lead_time_clip)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("target_name:N", title="Target", sort=["show", "no_show"]),
                y=alt.Y("lead_time_days:Q", title="Días de anticipación"),
                color=alt.Color("target_name:N", legend=None),
            )
            .properties(height=300, title="Lead time por clase")
        )
        st.altair_chart(box, use_container_width=True)
        medians = df.groupby("target_name")["lead_time_days"].median()
        st.caption(
            f"Mediana de lead time: show={medians.get('show', float('nan')):.0f} días vs. "
            f"no_show={medians.get('no_show', float('nan')):.0f} días. Es la señal más "
            "fuerte del dataset: a mayor anticipación, más chances de faltar."
        )
    with col2:
        rate_age = (
            df.groupby("age_group", observed=True)["no_show"]
            .mean()
            .reindex(AGE_GROUP_ORDER)
            .mul(100)
            .reset_index()
        )
        rate_age.columns = ["age_group", "pct"]
        chart_age = (
            alt.Chart(rate_age)
            .mark_bar()
            .encode(
                x=alt.X("age_group:N", title="Grupo etario", sort=AGE_GROUP_ORDER),
                y=alt.Y("pct:Q", title="Tasa de no-show (%)"),
                color=alt.Color("age_group:N", legend=None),
                tooltip=[alt.Tooltip("age_group:N"), alt.Tooltip("pct:Q", format=".1f")],
            )
            .properties(height=300, title="Tasa de no-show por grupo etario")
        )
        st.altair_chart(chart_age, use_container_width=True)
        st.caption(
            "Los adultos mayores faltan menos (turnos de control por "
            "comorbilidades, mayor disponibilidad de tiempo); menores y "
            "adultos en edad laboral, más."
        )

    col3, col4 = st.columns(2)
    with col3:
        rate_dow = df.groupby("appointment_dow")["no_show"].mean().mul(100).reset_index()
        rate_dow["dia"] = rate_dow["appointment_dow"].map(dict(enumerate(DOW_LABELS)))
        chart_dow = (
            alt.Chart(rate_dow)
            .mark_bar()
            .encode(
                x=alt.X("dia:N", title="Día del turno", sort=DOW_LABELS),
                y=alt.Y("no_show:Q", title="Tasa de no-show (%)"),
                tooltip=[alt.Tooltip("dia:N"), alt.Tooltip("no_show:Q", format=".1f")],
            )
            .properties(height=300, title="Tasa de no-show por día de la semana")
        )
        st.altair_chart(chart_dow, use_container_width=True)
    with col4:
        rate_rain = df.groupby("is_rainy")["no_show"].mean().mul(100).reset_index()
        rate_rain["clima"] = rate_rain["is_rainy"].map({0: "Sin lluvia", 1: "Con lluvia"})
        chart_rain = (
            alt.Chart(rate_rain)
            .mark_bar()
            .encode(
                x=alt.X("clima:N", title=""),
                y=alt.Y("no_show:Q", title="Tasa de no-show (%)"),
                color=alt.Color("clima:N", legend=None),
                tooltip=[alt.Tooltip("clima:N"), alt.Tooltip("no_show:Q", format=".1f")],
            )
            .properties(height=300, title="Clima del día del turno (fuente secundaria)")
        )
        st.altair_chart(chart_rain, use_container_width=True)
        st.caption(
            "El clima aporta una señal débil frente al comportamiento del "
            "paciente: es un hallazgo honesto, no un fracaso del pipeline."
        )

    with st.expander("⚠️ Caveat causal: SMS_received y lead time"):
        sms_rate = df.groupby("SMS_received")["no_show"].mean().mul(100)
        sms_lead = df.groupby("SMS_received")["lead_time_days"].median()
        st.write(
            f"Turnos **con SMS**: {sms_rate.get(1, float('nan')):.1f}% de no-show, "
            f"lead time mediano de {sms_lead.get(1, float('nan')):.0f} días.  \n"
            f"Turnos **sin SMS**: {sms_rate.get(0, float('nan')):.1f}% de no-show, "
            f"lead time mediano de {sms_lead.get(0, float('nan')):.0f} días."
        )
        st.markdown(
            "A simple vista el SMS 'aumenta' el no-show — pero el sistema solo "
            "envía SMS a turnos agendados con mucha anticipación, y ya vimos que "
            "el lead time por sí mismo se asocia a más ausencia. Es una "
            "correlación espuria (confounding), no una relación causal: aislar "
            "el efecto real del recordatorio requeriría un A/B test."
        )


# --- Tab 3: modelo -----------------------------------------------------------


def render_modelo(model, metrics: dict) -> None:
    st.subheader("Técnica de minería: clasificación con Árbol de Decisión y Random Forest")

    cv_results = metrics.get("cv_results", {})
    if cv_results:
        cv_df = pd.DataFrame(cv_results).T
        cv_df = cv_df.rename(
            columns={
                "f1_mean": "F1 (media CV)",
                "f1_std": "F1 (desvío CV)",
                "roc_auc_mean": "ROC-AUC (media CV)",
                "roc_auc_std": "ROC-AUC (desvío CV)",
            }
        )
        st.markdown("**Comparación por validación cruzada (5-fold estratificado, sobre train):**")
        st.dataframe(cv_df.style.format("{:.3f}"), use_container_width=True)
        st.caption(
            f"Modelo elegido por ROC-AUC de CV: **{metrics.get('best_model', '—')}** "
            "(más robusto que F1@0.5 bajo el desbalance ~80/20)."
        )

    fi_df = get_feature_importances(model, top_n=15)
    if not fi_df.empty:
        chart_fi = (
            alt.Chart(fi_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importancia"),
                y=alt.Y("feature:N", sort="-x", title=""),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=400, title="Importancia de variables (Random Forest)")
        )
        st.altair_chart(chart_fi, use_container_width=True)
        st.caption(
            "`lead_time_days` y `same_day` encabezan el ranking: confirma el "
            "storytelling del EDA con un criterio objetivo e independiente."
        )

    st.markdown("---")
    st.markdown("### Ajuste del umbral de decisión (en vivo)")
    st.caption(
        "Un Falso Negativo (no-show real no detectado) cuesta más que un Falso "
        "Positivo (recordatorio de más): por eso se prioriza recall sobre "
        "precision. Mové el umbral y mirá el trade-off en tiempo real."
    )

    X_test, y_test = load_test_split()
    y_proba = compute_test_proba(model)

    threshold = st.slider(
        "Umbral de decisión (probabilidad ≥ umbral ⇒ se predice no-show)",
        min_value=0.05,
        max_value=0.70,
        value=0.25,
        step=0.01,
    )

    single = threshold_analysis(y_test, y_proba, thresholds=[threshold]).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision (no_show)", f"{single['precision']:.1%}")
    col2.metric("Recall (no_show)", f"{single['recall']:.1%}")
    col3.metric("F1 (no_show)", f"{single['f1']:.3f}")
    col4.metric(
        "Turnos marcados de riesgo",
        f"{int((y_proba >= threshold).sum()):,}",
        help="Sobre los 22.105 turnos del hold-out de test",
    )

    col_cm, col_curve = st.columns(2)
    with col_cm:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        labels = ["show", "no_show"]
        cm_rows = [
            {"Real": labels[i], "Predicho": labels[j], "Cantidad": int(cm[i, j])}
            for i in range(2)
            for j in range(2)
        ]
        cm_df = pd.DataFrame(cm_rows)
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicho:N", sort=labels),
            y=alt.Y("Real:N", sort=labels),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Cantidad:Q", scale=alt.Scale(scheme="blues"), legend=None)
        )
        text = base.mark_text(baseline="middle", fontSize=16).encode(
            text="Cantidad:Q",
            color=alt.condition(
                alt.datum.Cantidad > float(cm.max()) / 2, alt.value("white"), alt.value("black")
            ),
        )
        st.altair_chart(
            (heat + text).properties(height=280, title=f"Matriz de confusión @ umbral={threshold:.2f}"),
            use_container_width=True,
        )
    with col_curve:
        grid = [round(t, 2) for t in np.arange(0.05, 0.71, 0.01)]
        table = threshold_analysis(y_test, y_proba, thresholds=grid)
        long = table.melt(
            id_vars="threshold", value_vars=["precision", "recall", "f1"],
            var_name="métrica", value_name="valor",
        )
        line = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("threshold:Q", title="Umbral de decisión"),
                y=alt.Y("valor:Q", title="Métrica (clase no_show)"),
                color=alt.Color("métrica:N"),
                tooltip=["métrica", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("valor:Q", format=".3f")],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"threshold": [threshold]}))
            .mark_rule(color="firebrick", strokeDash=[4, 4])
            .encode(x="threshold:Q")
        )
        st.altair_chart(
            (line + rule).properties(height=280, title="Precision / recall / F1 según umbral"),
            use_container_width=True,
        )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    best_model_name = metrics.get("best_model", "random_forest")
    auc = metrics.get("test_metrics", {}).get(best_model_name, {}).get("roc_auc")
    roc_line = alt.Chart(roc_df).mark_line().encode(x="fpr:Q", y="tpr:Q")
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q", y="y:Q")
    with st.expander(f"Curva ROC (AUC = {auc:.3f})" if auc else "Curva ROC"):
        st.altair_chart(
            (roc_line + diag).properties(
                height=320, title="Curva ROC — hold-out de test"
            ).interactive(),
            use_container_width=True,
        )

    tree_path = config.ROOT_DIR / "reports" / "figures" / "decision_tree.png"
    if tree_path.exists():
        with st.expander("Árbol de Decisión podado (interpretabilidad)"):
            st.image(str(tree_path), use_container_width=True)


# --- Tab 4: conclusión -------------------------------------------------------


def render_conclusion(metrics: dict) -> None:
    st.subheader("Conclusión")
    best_model = metrics.get("best_model", "random_forest")
    roc_auc = metrics.get("test_metrics", {}).get(best_model, {}).get("roc_auc")

    st.markdown(
        f"""
La hipótesis de trabajo sostenía que el no-show no es aleatorio y que
variables observables al agendar el turno permiten estimarlo con una
capacidad de discriminación útil para el negocio. El resultado la
sostiene: **ROC-AUC de test ≈ {roc_auc:.3f}** (muy por encima del azar,
0,5), y el ranking de importancia de variables coincide con lo hallado en
el EDA (`lead_time_days`/`same_day` como señal principal).

**Valor de negocio:** con el umbral orientado a recall (~0,25), el modelo
detecta ≈72% de los no-show reales del hold-out de test, permitiendo
accionar (recordatorio o sobreturno) antes de que la franja quede vacía
sin aviso — cada no-show anticipado y compensado recupera una
hora-profesional. La app (modo *Turno individual* / *Agenda del día*)
traduce esa probabilidad en la acción concreta.

**Limitaciones:** datos de 2016, una única ciudad (Vitória, ES, Brasil) y
pre-COVID; posible *concept drift* que exige reentrenar periódicamente;
ventana de clima acotada (~1,5 meses); el efecto de `SMS_received` está
confundido con el lead time (no interpretable causalmente sin un diseño
experimental); es un modelo académico (TPO), sin monitoreo continuo en
producción.
        """
    )


def render_dashboard(model) -> None:
    df = load_processed_dataset()
    metrics = load_metrics()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Contexto", "🔎 Análisis exploratorio", "🌳 Modelo", "✅ Conclusión"]
    )
    with tab1:
        render_contexto(df, metrics)
    with tab2:
        render_eda(df)
    with tab3:
        render_modelo(model, metrics)
    with tab4:
        render_conclusion(metrics)
