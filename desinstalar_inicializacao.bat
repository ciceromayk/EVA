@echo off
REM Remove a inicializacao automatica do servidor da EVA.
set LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\EVA-Servidor.lnk
if exist "%LNK%" (
    del "%LNK%"
    echo Inicializacao automatica removida.
) else (
    echo Nenhuma inicializacao automatica encontrada.
)
pause
