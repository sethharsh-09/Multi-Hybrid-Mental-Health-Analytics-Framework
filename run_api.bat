@echo off
title MindGraphNet - FastAPI Microservice
echo ========================================================
echo  Starting MindGraphNet FastAPI Microservice
echo ========================================================
set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

if exist "%~dp0..\river water analytics system\.venv\Scripts\uvicorn.exe" (
    "%~dp0..\river water analytics system\.venv\Scripts\uvicorn.exe" api.app:app --host 127.0.0.1 --port 8000 --reload
    goto end
)
if exist "%~dp0.venv\Scripts\uvicorn.exe" (
    "%~dp0.venv\Scripts\uvicorn.exe" api.app:app --host 127.0.0.1 --port 8000 --reload
    goto end
)

uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

:end
pause
