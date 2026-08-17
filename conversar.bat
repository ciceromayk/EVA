@echo off
REM Atalho para conversar com a EVA no Windows (clique duplo).
cd /d "%~dp0"
REM Limpa o PYTHONPATH desta sessao para nao herdar pacotes de outro Python.
set "PYTHONPATH="
call "%~dp0localizar_python.bat"
if not defined PY (pause & exit /b 1)
echo Instalando dependencias (demora so na primeira vez)...
%PY% -E -m pip install -q numpy
echo.
echo Abrindo a Sala de Conversa da EVA em http://localhost:8001
start "" http://localhost:8001
%PY% -E chat.py
pause
