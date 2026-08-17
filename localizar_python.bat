@echo off
REM Script INTERNO: descobre qual comando Python usar nesta maquina e
REM guarda em %PY%. Chamado com "call" pelos outros atalhos (iniciar.bat,
REM conversar.bat, servidor.bat, configurar_gpu.bat) -- nao precisa
REM rodar este arquivo diretamente.
REM
REM Preferencia: Python 3.12 (versao testada do projeto) -> qualquer
REM Python 3 do "py launcher" -> "py" padrao -> "python" no PATH.
set "PY="

py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"

if not defined PY (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    py --version >nul 2>&1
    if not errorlevel 1 set "PY=py"
)

if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo.
    echo [ERRO] Nenhum Python encontrado nesta maquina.
    echo Baixe e instale o Python em https://www.python.org/downloads/
    echo ^(marque a opcao "Add python.exe to PATH" durante a instalacao^)
    echo e rode este atalho de novo.
    echo.
    exit /b 1
)

exit /b 0
