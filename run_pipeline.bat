@echo off
title MindGraphNet - Multi-Hybrid Pipeline Execution
echo ========================================================
echo  Executing MindGraphNet 9-Stage ML Pipeline
echo ========================================================
set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

if exist "%~dp0..\river water analytics system\.venv\Scripts\python.exe" (
    "%~dp0..\river water analytics system\.venv\Scripts\python.exe" "%~dp0MindGraphNet_Enhanced_Pipeline.py"
    goto end
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0MindGraphNet_Enhanced_Pipeline.py"
    goto end
)

python "%~dp0MindGraphNet_Enhanced_Pipeline.py"

:end
pause
