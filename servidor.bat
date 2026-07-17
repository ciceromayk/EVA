@echo off
REM Sobe a EVA como servidor protegido por senha, pronto para acesso remoto
REM via Tailscale. Use este atalho (em vez do iniciar.bat) quando quiser
REM treinar de outro aparelho.
cd /d "%~dp0"

if not exist "senha_servidor.txt" (
    echo Primeira vez rodando o servidor: gerando uma senha de acesso...
    py -3.12 -c "import secrets; print(secrets.token_urlsafe(9))" > senha_servidor.txt
)

set /p EVA_PASSWORD=<senha_servidor.txt

echo ============================================
echo  EVA - Servidor
echo ============================================
echo  Usuario : eva
echo  Senha   : %EVA_PASSWORD%
echo  (a senha tambem fica salva em senha_servidor.txt)
echo ============================================
echo.
echo Instalando dependencias (so na primeira vez)...
py -3.12 -m pip install -q numpy pymupdf

echo.
echo Descobrindo o endereco Tailscale desta maquina...
tailscale ip -4 2>nul
echo (se nao aparecer nada acima, veja o README - secao Tailscale)
echo.
echo Painel local: http://localhost:8000
echo.
py -3.12 -E app.py
pause
