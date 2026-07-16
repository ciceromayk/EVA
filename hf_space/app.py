"""Bootstrap para o Hugging Face Spaces (SDK Gradio — gratuito).

Cole os três arquivos desta pasta (app.py, requirements.txt, README.md)
num Space Gradio em branco. Ao subir, este script baixa o código da EVA do
GitHub e abre a interface Gradio — nada mais a configurar.
"""

import io
import os
import sys
import tarfile
import urllib.request

BRANCH = "claude/custom-ai-from-scratch-dx8d5p"
SRC = "eva_src"

# Baixa o repositório como tarball (não depende de git estar instalado).
if not os.path.isdir(SRC):
    url = "https://codeload.github.com/ciceromayk/EVA/tar.gz/refs/heads/" + BRANCH
    raw = urllib.request.urlopen(url).read()
    tf = tarfile.open(fileobj=io.BytesIO(raw))
    top = tf.getnames()[0].split("/")[0]
    tf.extractall(".")
    os.rename(top, SRC)

sys.path.insert(0, os.path.abspath(SRC))
os.chdir(SRC)

import gradio_ui  # noqa: E402

gradio_ui.launch_demo(server_name="0.0.0.0", server_port=7860)
