@echo off
REM Atalho para iniciar o painel da EVA no Windows (clique duplo).
cd /d "%~dp0"
echo Instalando dependencias (demora so na primeira vez)...
py -m pip install -q numpy pymupdf
echo.
echo Abrindo o painel da EVA em http://localhost:8000
start "" http://localhost:8000
py app.py
pause
