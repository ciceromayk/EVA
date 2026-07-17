@echo off
REM Faz o servidor da EVA iniciar sozinho quando o Windows liga, sem
REM precisar abrir nada manualmente. Rode este atalho UMA VEZ.
cd /d "%~dp0"

set TARGET=%~dp0servidor.bat
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set LNK=%STARTUP%\EVA-Servidor.lnk

powershell -NoProfile -Command ^
  "$s = (New-Object -COM WScript.Shell).CreateShortcut('%LNK%');" ^
  "$s.TargetPath = '%TARGET%';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'Servidor da EVA';" ^
  "$s.Save()"

if exist "%LNK%" (
    echo Pronto! O servidor da EVA agora inicia sozinho quando o Windows liga.
    echo Atalho criado em: %LNK%
    echo.
    echo Para desativar depois, apague esse atalho ou rode desinstalar_inicializacao.bat
) else (
    echo Nao foi possivel criar o atalho. Rode este arquivo como administrador,
    echo ou adicione manualmente um atalho de servidor.bat na pasta:
    echo   %STARTUP%
)
pause
