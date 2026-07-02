#!/usr/bin/env bash
#
# setup.sh — inicializa el proyecto no-show de punta a punta (pasos 2 a 5 del README):
#   2. crea el entorno virtual (Python 3.13) e instala dependencias
#   3. verifica que el dataset de turnos esté en data/raw/
#   4. entrena el modelo (genera models/model.joblib)
#   5. levanta la app Streamlit
#
# Uso:
#   ./setup.sh              # hace 2→3→4→5 (la app queda corriendo en foreground)
#   ./setup.sh --no-app     # hace 2→3→4 y termina (no lanza Streamlit)
#   FORCE_TRAIN=1 ./setup.sh   # reentrena aunque ya exista models/model.joblib
#
set -euo pipefail
cd "$(dirname "$0")"

RUN_APP=1
[[ "${1:-}" == "--no-app" ]] && RUN_APP=0

echo "==> [2/5] Entorno virtual e instalación de dependencias"
# Elegir intérprete: preferimos 3.13 (3.14 puede no tener wheels científicos)
PY=""
for c in python3.13 python3.12 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: no se encontró Python 3. Instalá Python 3.13." >&2
  exit 1
fi
echo "    usando $($PY --version 2>&1) ($PY)"

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

echo "==> [3/5] Verificando datos crudos"
if [[ ! -f data/raw/no-show-dataset.csv ]]; then
  echo "ERROR: falta data/raw/no-show-dataset.csv" >&2
  echo "       Descargá el dataset 'Medical Appointment No Shows' de Kaggle y ponelo ahí." >&2
  exit 1
fi
echo "    data/raw/no-show-dataset.csv OK"
if [[ ! -f data/raw/weather-dataset.csv && ! -f data/external/weather_daily_a612.csv ]]; then
  echo "ERROR: falta el clima. Poné data/raw/weather-dataset.csv (INMET 2016) o el caché" >&2
  echo "       data/external/weather_daily_a612.csv (versionado en el repo)." >&2
  exit 1
fi

echo "==> [4/5] Entrenando el modelo"
if [[ -f models/model.joblib && "${FORCE_TRAIN:-0}" != "1" ]]; then
  echo "    models/model.joblib ya existe (usá FORCE_TRAIN=1 para reentrenar) — salteando"
else
  python -m noshow.train
fi

if [[ "$RUN_APP" == "0" ]]; then
  echo "==> Listo (2→4). Para levantar la app: streamlit run app/streamlit_app.py"
  exit 0
fi

echo "==> [5/5] Levantando la app Streamlit (Ctrl+C para cortar)"
exec streamlit run app/streamlit_app.py
