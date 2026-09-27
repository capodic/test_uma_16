\@echo off
REM Avvio UMA-16 Drone Locator (apre il browser su http://localhost:5006/)
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
python main.py
pause
