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
echo Instalando as bibliotecas de runtime da CUDA (curand, cublas, etc.)...
py -3.12 -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-nvrtc-cu12 nvidia-cuda-nvrtc-cu12
echo.
echo ============================================
echo  Testando a GPU...
echo ============================================
py -3.12 check_gpu.py
echo.
pause
