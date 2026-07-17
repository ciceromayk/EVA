@echo off
REM Baixa a versao mais recente da EVA do GitHub (clique duplo).
REM Requer que a pasta tenha sido obtida com "git clone" (nao pelo .zip).
cd /d "%~dp0"
echo Baixando as ultimas atualizacoes da EVA...
echo.
git pull
echo.
echo ---------------------------------------------
echo Pronto! Seus materiais e modelos treinados nao sao afetados.
echo Agora feche esta janela e rode o iniciar.bat de novo.
echo ---------------------------------------------
pause
