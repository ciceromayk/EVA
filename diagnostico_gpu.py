"""Diagnóstico detalhado da GPU — mostra o erro cru do CuPy e o que existe.

Uso:  py -3.12 diagnostico_gpu.py
Cole a saída inteira no chat para o diagnóstico.
"""

import os
import sys

print("=" * 60)
print("Python:", sys.version.split()[0], "|", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH", "(vazio)"))
# Mostra de onde o CuPy vai ser carregado — revela mistura de versoes
try:
    import cupy as _c
    print("cupy carregado de:", os.path.dirname(_c.__file__))
except Exception as _e:
    print("cupy ainda nao importavel:", type(_e).__name__)
print("=" * 60)

# 1) Os pacotes nvidia-* foram instalados? Onde estão as DLLs?
print("\n[1] Pacotes NVIDIA e DLLs encontradas:")
try:
    import nvidia
    found_dll = False
    for root in nvidia.__path__:
        for lib in sorted(os.listdir(root)):
            binp = os.path.join(root, lib, "bin")
            if os.path.isdir(binp):
                dlls = [f for f in os.listdir(binp) if f.lower().endswith(".dll")]
                print(f"  {lib}/bin -> {len(dlls)} dll(s)")
                for d in dlls:
                    if "curand" in d.lower() or "cublas" in d.lower():
                        found_dll = True
    if not found_dll:
        print("  (nenhuma curand/cublas encontrada!)")
except Exception as e:
    print("  Pacote 'nvidia' NAO encontrado:", e)
    print("  -> os wheels nvidia-*-cu12 nao foram instalados.")

# 2) Registra as pastas de DLL (como o backend da EVA faz)
print("\n[2] Registrando pastas de DLL...")
if hasattr(os, "add_dll_directory"):
    try:
        import nvidia
        for root in nvidia.__path__:
            for lib in os.listdir(root):
                binp = os.path.join(root, lib, "bin")
                if os.path.isdir(binp):
                    os.add_dll_directory(binp)
        print("  ok")
    except Exception as e:
        print("  falhou:", e)

# 3) Tenta importar o CuPy e usar a GPU — mostrando o erro COMPLETO
print("\n[3] Testando o CuPy (erro completo, se houver):")
try:
    import cupy as cp
    print("  cupy versao:", cp.__version__)
    print("  GPUs visiveis:", cp.cuda.runtime.getDeviceCount())
    a = cp.random.random(4).astype(cp.float32)
    print("  random+matmul na GPU:", float((a @ a).sum()))
    print("\n  >>> GPU FUNCIONANDO! <<<")
except Exception:
    import traceback
    traceback.print_exc()
    print("\n  >>> GPU ainda com problema (veja o traceback acima) <<<")

# 4) Driver NVIDIA
print("\n[4] Driver NVIDIA (nvidia-smi):")
os.system("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
