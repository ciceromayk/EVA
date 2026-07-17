@echo off
REM Instala a EVA + suporte a GPU (CuPy) no Python 3.12 e testa a placa.
REM Use este atalho uma vez para preparar a GPU.
cd /d "%~dp0"
echo ============================================
echo  Instalando EVA + GPU no Python 3.12...
echo  (a primeira vez baixa varios MB, tenha paciencia)
echo ============================================
echo.
py -3.12 -m pip install numpy pymupdf cupy-cuda12x
echo.
echo ============================================
echo  Testando a GPU...
echo ============================================
py -3.12 check_gpu.py
echo.
pause
