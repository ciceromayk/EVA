# Empacota o painel da EVA num container — roda em qualquer nuvem que aceite
# Docker (Hugging Face Spaces, Render, Railway, Fly.io, Koyeb, etc.).
FROM python:3.11-slim

WORKDIR /app

# Dependências primeiro (aproveita o cache de camadas do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código da EVA
COPY . .

# O Hugging Face Spaces serve na porta 7860; outros hosts injetam $PORT,
# que o app.py lê automaticamente (padrão 7860 aqui).
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
