@echo off
REM Cria atalhos da EVA na sua area de trabalho (clique duplo, uma vez so):
REM   "EVA - Conversa"  abre a Sala de Conversa (usar o modelo)
REM   "EVA - Painel"    abre o painel de estudo (alimentar e treinar)
cd /d "%~dp0"

REM A area de trabalho real pode estar no OneDrive; o PowerShell descobre.
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set DESKTOP=%%D

set LNK1=%DESKTOP%\EVA - Conversa.lnk
set LNK2=%DESKTOP%\EVA - Painel.lnk

powershell -NoProfile -Command ^
  "$s = (New-Object -COM WScript.Shell).CreateShortcut('%LNK1%');" ^
  "$s.TargetPath = '%~dp0conversar.bat';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.Description = 'Conversar com a EVA (Sala de Conversa)';" ^
  "$s.IconLocation = 'shell32.dll,25';" ^
  "$s.Save()"

powershell -NoProfile -Command ^
  "$s = (New-Object -COM WScript.Shell).CreateShortcut('%LNK2%');" ^
  "$s.TargetPath = '%~dp0iniciar.bat';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.Description = 'Painel de estudo da EVA (alimentar e treinar)';" ^
  "$s.IconLocation = 'shell32.dll,13';" ^
  "$s.Save()"

echo.
if exist "%LNK1%" (
    echo Pronto! Atalhos criados na sua area de trabalho:
    echo   EVA - Conversa   ^(usar o modelo, http://localhost:8001^)
    echo   EVA - Painel     ^(alimentar e treinar, http://localhost:8000^)
) else (
    echo Nao foi possivel criar os atalhos automaticamente.
    echo Crie manualmente: botao direito na area de trabalho ^> Novo ^> Atalho,
    echo apontando para conversar.bat ou iniciar.bat desta pasta:
    echo   %~dp0
)
echo.
pause
