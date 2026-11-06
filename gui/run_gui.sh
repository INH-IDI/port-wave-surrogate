#!/bin/bash
# Launch Streamlit GUI for PortWaveSurrogate
# Uses python from ML venv (which includes streamlit) or fallback to pyenv

ML_PYTHON="/home/model2/ambientes/ML/bin/python3"
if [ -f "$ML_PYTHON" ]; then
    PYTHON_CMD="$ML_PYTHON"
else
    PYTHON_CMD="python3"
fi

GUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting PortWaveSurrogate Dashboard on: http://localhost:8501"
"$PYTHON_CMD" -m streamlit run "$GUI_DIR/app.py" "$@"
