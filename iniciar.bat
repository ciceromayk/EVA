@echo off
REM Atalho para iniciar o painel da EVA no Windows (clique duplo).
cd /d "%~dp0"
REM Limpa o PYTHONPATH desta sessao para nao herdar pacotes de outro Python.
set "PYTHONPATH="
call "%~dp0localizar_python.bat"
if not defined PY (pause & exit /b 1)
echo Instalando dependencias (demora so na primeira vez)...
%PY% -E -m pip install -q numpy pymupdf
echo.
echo Abrindo o painel da EVA em http://localhost:8000
start "" http://localhost:8000
%PY% -E app.py
pause
