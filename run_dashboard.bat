@echo off
title MindGraphNet - Clinical Analytics Dashboard
echo ========================================================
echo  Launching MindGraphNet Interactive Streamlit Portal
echo ========================================================
set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

set PY_EXE=python
if exist "%~dp0..\river water analytics system\.venv\Scripts\streamlit.exe" (
    "%~dp0..\river water analytics system\.venv\Scripts\streamlit.exe" run "%~dp0dashboard\streamlit_app.py" --server.port 8501
    goto end
)
if exist "%~dp0.venv\Scripts\streamlit.exe" (
    "%~dp0.venv\Scripts\streamlit.exe" run "%~dp0dashboard\streamlit_app.py" --server.port 8501
    goto end
)

streamlit run "%~dp0dashboard\streamlit_app.py" --server.port 8501

:end
pause
