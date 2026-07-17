@echo off
REM Atalho para iniciar o painel da EVA no Windows (clique duplo).
cd /d "%~dp0"
REM Limpa o PYTHONPATH desta sessao para nao herdar pacotes de outro Python.
set "PYTHONPATH="
echo Instalando dependencias (demora so na primeira vez)...
py -3.12 -E -m pip install -q numpy pymupdf
echo.
echo Abrindo o painel da EVA em http://localhost:8000
start "" http://localhost:8000
py -3.12 -E app.py
pause
