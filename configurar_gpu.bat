@echo off
REM Instala a EVA + suporte a GPU (CuPy) e testa a placa.
REM Use este atalho uma vez para preparar a GPU.
cd /d "%~dp0"
REM Limpa o PYTHONPATH desta sessao para nao herdar pacotes de outro Python.
set "PYTHONPATH="
call "%~dp0localizar_python.bat"
if not defined PY (pause & exit /b 1)
echo ============================================
echo  Instalando EVA + GPU...
echo  (Python 3.12 e o mais testado para os pacotes CUDA; se sua maquina
echo  tiver outra versao, a instalacao ainda tenta, mas fique de olho
echo  em erros de wheel/compatibilidade abaixo)
echo  (a primeira vez baixa varios MB, tenha paciencia)
echo ============================================
echo.
%PY% -E -m pip install numpy pymupdf cupy-cuda12x
echo.
echo Instalando as bibliotecas de runtime da CUDA (curand, cublas, etc.)...
%PY% -E -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
echo.
echo ============================================
echo  Testando a GPU...
echo ============================================
%PY% -E check_gpu.py
echo.
pause
