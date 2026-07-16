"""Painel web da EVA — estude o modelo com seu próprio material.

Uma interface local e autossuficiente (só usa a biblioteca padrão do
Python) para você, sem depender de ninguém:

  1. adicionar material (PDF ou texto) ao corpus,
  2. treinar a EVA nesse material,
  3. gerar texto com o modelo treinado.

Uso:
    pip install numpy pymupdf
    python app.py            # abre em http://localhost:8000

O material que você envia fica em `materials/`, e o corpus (`data/corpus.txt`)
é reconstruído a partir dele. O treino e a geração reaproveitam o `train.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
MATERIALS = os.path.join(BASE, "materials")
CORPUS = os.path.join(BASE, "data", "corpus.txt")
LOG = os.path.join(BASE, "train_run.log")
CKPT = os.path.join(BASE, "eva_checkpoint.pkl")
ALLOWED_EXT = (".pdf", ".txt", ".md")

sys.path.insert(0, BASE)
from tools.build_corpus import build  # noqa: E402

# Estado do processo de treino em andamento (um por vez).
_train_proc: subprocess.Popen | None = None
_lock = threading.Lock()


# ----------------------------------------------------------------------
# Operações de corpus / materiais
# ----------------------------------------------------------------------
def list_materials() -> list[dict]:
    os.makedirs(MATERIALS, exist_ok=True)
    items = []
    for name in sorted(os.listdir(MATERIALS)):
        path = os.path.join(MATERIALS, name)
        if os.path.isfile(path):
            items.append({"name": name, "size": os.path.getsize(path)})
    return items


def rebuild_corpus() -> dict:
    """Reconstrói data/corpus.txt a partir de tudo em materials/."""
    os.makedirs(MATERIALS, exist_ok=True)
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    paths = [os.path.join(MATERIALS, m["name"]) for m in list_materials()]
    text = build(paths) if paths else ""
    with open(CORPUS, "w", encoding="utf-8") as f:
        f.write(text)
    return corpus_stats()


def corpus_stats() -> dict:
    if not os.path.exists(CORPUS):
        return {"chars": 0, "words": 0, "vocab": 0}
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    return {"chars": len(text), "words": len(text.split()), "vocab": len(set(text))}


def safe_name(name: str) -> str:
    """Evita path traversal: fica só com o nome-base e extensões conhecidas."""
    name = os.path.basename(name).replace("\\", "_").replace("/", "_")
    return name or "arquivo.txt"


# ----------------------------------------------------------------------
# Treino / geração (reaproveitam train.py como subprocesso)
# ----------------------------------------------------------------------
def training_active() -> bool:
    return _train_proc is not None and _train_proc.poll() is None


def start_training(opts: dict) -> tuple[bool, str]:
    global _train_proc
    with _lock:
        if training_active():
            return False, "Já existe um treino em andamento."
        if corpus_stats()["chars"] < 200:
            return False, "Corpus muito pequeno. Adicione material primeiro."
        cmd = [
            sys.executable, os.path.join(BASE, "train.py"),
            "--preset", str(opts.get("preset", "medium")),
            "--tokenizer", str(opts.get("tokenizer", "char")),
            "--steps", str(int(opts.get("steps", 1500))),
            "--dropout", str(float(opts.get("dropout", 0.1))),
            "--log-every", "25",
        ]
        if opts.get("tokenizer") == "bpe":
            cmd += ["--bpe-vocab", str(int(opts.get("bpe_vocab", 512)))]
        logfile = open(LOG, "w")
        _train_proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                       cwd=BASE, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return True, "Treino iniciado."


def stop_training() -> tuple[bool, str]:
    with _lock:
        if not training_active():
            return False, "Nenhum treino em andamento."
        _train_proc.terminate()
    return True, "Treino interrompido."


def read_log() -> str:
    if not os.path.exists(LOG):
        return ""
    with open(LOG, encoding="utf-8", errors="replace") as f:
        return f.read()


def generate(prompt: str, max_tokens: int) -> str:
    if not os.path.exists(CKPT):
        return "(sem checkpoint — treine a EVA primeiro)"
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(BASE, "train.py"),
             "--generate", prompt or "A ", "--max-new", str(max_tokens)],
            cwd=BASE, capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except subprocess.TimeoutExpired:
        return "(a geração demorou demais)"
    text = out.stdout
    # extrai o trecho entre os marcadores da amostra
    start = text.find("--- amostra gerada ---")
    end = text.rfind("----------------------")
    if start != -1 and end != -1 and end > start:
        return text[start + len("--- amostra gerada ---"):end].strip()
    return (text + out.stderr).strip() or "(sem saída)"


# ----------------------------------------------------------------------
# Servidor HTTP
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencia o log padrão no terminal
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ------------------------------------------------------------------
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/status":
            self._json({
                "corpus": corpus_stats(),
                "materials": list_materials(),
                "training": training_active(),
                "has_checkpoint": os.path.exists(CKPT),
            })
        elif path == "/log":
            self._send(200, read_log(), "text/plain; charset=utf-8")
        else:
            self._send(404, "não encontrado", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            if path == "/upload":
                name = safe_name(qs.get("name", ["arquivo.txt"])[0])
                if not name.lower().endswith(ALLOWED_EXT):
                    return self._json({"ok": False, "msg": "Use .pdf, .txt ou .md"}, 400)
                os.makedirs(MATERIALS, exist_ok=True)
                with open(os.path.join(MATERIALS, name), "wb") as f:
                    f.write(self._body())
                stats = rebuild_corpus()
                self._json({"ok": True, "msg": f"'{name}' adicionado.", "corpus": stats})

            elif path == "/add_text":
                data = json.loads(self._body() or b"{}")
                text = (data.get("text") or "").strip()
                name = safe_name(data.get("name") or "texto_colado.txt")
                if not name.lower().endswith((".txt", ".md")):
                    name += ".txt"
                if len(text) < 10:
                    return self._json({"ok": False, "msg": "Texto muito curto."}, 400)
                os.makedirs(MATERIALS, exist_ok=True)
                with open(os.path.join(MATERIALS, name), "w", encoding="utf-8") as f:
                    f.write(text)
                self._json({"ok": True, "msg": f"'{name}' adicionado.",
                            "corpus": rebuild_corpus()})

            elif path == "/remove":
                data = json.loads(self._body() or b"{}")
                name = safe_name(data.get("name", ""))
                target = os.path.join(MATERIALS, name)
                if os.path.exists(target):
                    os.remove(target)
                self._json({"ok": True, "msg": f"'{name}' removido.",
                            "corpus": rebuild_corpus()})

            elif path == "/train":
                ok, msg = start_training(json.loads(self._body() or b"{}"))
                self._json({"ok": ok, "msg": msg})

            elif path == "/stop":
                ok, msg = stop_training()
                self._json({"ok": ok, "msg": msg})

            elif path == "/generate":
                data = json.loads(self._body() or b"{}")
                result = generate(data.get("prompt", ""),
                                  int(data.get("max_tokens", 250)))
                self._json({"ok": True, "text": result})
            else:
                self._json({"ok": False, "msg": "rota desconhecida"}, 404)
        except Exception as exc:  # devolve o erro em vez de derrubar o servidor
            self._json({"ok": False, "msg": f"erro: {exc}"}, 500)


PAGE = r"""<!doctype html>
<html lang="pt-br"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVA — Painel de Estudo</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       background: #0f1216; color: #e7ecf2; line-height: 1.5; }
header { padding: 22px 28px; border-bottom: 1px solid #232a33;
         background: linear-gradient(90deg,#161b22,#0f1216); }
h1 { margin: 0; font-size: 20px; letter-spacing: .5px; }
h1 small { color: #8aa0b4; font-weight: 400; font-size: 13px; }
main { max-width: 940px; margin: 0 auto; padding: 24px; display: grid; gap: 20px; }
.card { background: #161b22; border: 1px solid #232a33; border-radius: 12px; padding: 20px; }
.card h2 { margin: 0 0 14px; font-size: 15px; color: #cdd8e3; display:flex; align-items:center; gap:8px; }
.stats { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 12px; }
.stat { background:#0f141a; border:1px solid #232a33; border-radius:9px; padding:10px 16px; min-width:110px; }
.stat b { display:block; font-size:22px; color:#5fb0ff; } .stat span{ font-size:12px; color:#8aa0b4; }
.mat { display:flex; justify-content:space-between; align-items:center; padding:8px 12px;
       background:#0f141a; border:1px solid #232a33; border-radius:8px; margin-bottom:6px; font-size:14px; }
.mat button { background:none; border:none; color:#ff6b6b; cursor:pointer; font-size:16px; }
button.primary { background:#2563eb; color:#fff; border:none; border-radius:8px; padding:10px 18px;
                 font-size:14px; cursor:pointer; font-weight:600; }
button.primary:hover { background:#1d4ed8; } button:disabled{ opacity:.5; cursor:not-allowed; }
button.ghost { background:#0f141a; color:#e7ecf2; border:1px solid #333c47; border-radius:8px;
               padding:9px 16px; cursor:pointer; font-size:14px; }
label { font-size:13px; color:#9fb1c2; display:block; margin:10px 0 4px; }
input, select, textarea { width:100%; background:#0f141a; border:1px solid #2b333d; color:#e7ecf2;
                          border-radius:8px; padding:9px 11px; font-size:14px; font-family:inherit; }
textarea { resize:vertical; }
.row { display:flex; gap:14px; flex-wrap:wrap; } .row > div{ flex:1; min-width:130px; }
.drop { border:2px dashed #33404d; border-radius:10px; padding:26px; text-align:center; color:#8aa0b4;
        cursor:pointer; transition:.15s; } .drop.over{ border-color:#2563eb; background:#0f1a2e; color:#cfe0ff; }
pre { background:#0b0e12; border:1px solid #232a33; border-radius:8px; padding:14px; overflow:auto;
      max-height:320px; font-size:12.5px; white-space:pre-wrap; }
.msg { font-size:13px; margin-top:10px; min-height:18px; }
.ok{ color:#4ade80; } .err{ color:#ff6b6b; }
.badge{ font-size:11px; padding:2px 9px; border-radius:20px; background:#233; color:#8aa0b4; }
.badge.live{ background:#14361f; color:#4ade80; }
.out{ background:#0b0e12; border:1px solid #232a33; border-radius:8px; padding:16px; min-height:60px;
      font-size:14.5px; white-space:pre-wrap; }
.hint{ font-size:12px; color:#7d8ea0; margin-top:6px; }
</style></head>
<body>
<header><h1>EVA · Painel de Estudo <small>— alimente, treine e converse com sua IA</small></h1></header>
<main>

  <div class="card">
    <h2>📚 Corpus atual</h2>
    <div class="stats" id="stats">
      <div class="stat"><b id="s-chars">–</b><span>caracteres</span></div>
      <div class="stat"><b id="s-words">–</b><span>palavras</span></div>
      <div class="stat"><b id="s-vocab">–</b><span>vocabulário</span></div>
      <div class="stat"><b id="s-mats">–</b><span>materiais</span></div>
    </div>
    <div id="matlist"></div>
  </div>

  <div class="card">
    <h2>➕ Adicionar material</h2>
    <div class="drop" id="drop">Arraste PDFs/TXT aqui ou <b>clique para escolher</b>
      <input type="file" id="file" multiple accept=".pdf,.txt,.md" style="display:none">
    </div>
    <label>…ou cole um texto</label>
    <textarea id="paste" rows="4" placeholder="Cole aqui qualquer texto para a EVA estudar…"></textarea>
    <div style="margin-top:10px"><button class="ghost" id="addtext">Adicionar texto colado</button></div>
    <div class="msg" id="add-msg"></div>
  </div>

  <div class="card">
    <h2>🧠 Treinar <span class="badge" id="train-badge">ocioso</span></h2>
    <div class="row">
      <div><label>Tamanho do modelo</label><select id="preset">
        <option value="small">small (~350k, rápido)</option>
        <option value="medium" selected>medium (~1,8M)</option>
        <option value="large">large (~4,8M, lento)</option></select></div>
      <div><label>Tokenizador</label><select id="tokenizer">
        <option value="char">char (letra a letra)</option>
        <option value="bpe">bpe (subpalavras)</option></select></div>
      <div><label>Passos</label><input type="number" id="steps" value="1500" min="100" step="100"></div>
      <div><label>Dropout</label><input type="number" id="dropout" value="0.1" min="0" max="0.9" step="0.05"></div>
    </div>
    <div style="margin-top:16px; display:flex; gap:10px">
      <button class="primary" id="btn-train">Treinar EVA</button>
      <button class="ghost" id="btn-stop">Parar</button>
    </div>
    <div class="msg" id="train-msg"></div>
    <label style="margin-top:14px">Progresso</label>
    <pre id="log">(o log do treino aparece aqui)</pre>
  </div>

  <div class="card">
    <h2>💬 Gerar texto</h2>
    <label>Início do texto (prompt)</label>
    <input id="prompt" value="A arte da guerra" placeholder="Comece uma frase…">
    <div class="row" style="margin-top:10px; align-items:end">
      <div><label>Tokens a gerar</label><input type="number" id="maxtok" value="250" min="20" max="800" step="10"></div>
      <div style="flex:0"><button class="primary" id="btn-gen">Gerar</button></div>
    </div>
    <div class="hint" id="gen-hint">Requer um treino concluído.</div>
    <div class="out" id="out" style="margin-top:12px"></div>
  </div>

</main>
<script>
const $ = s => document.querySelector(s);
const fmt = n => n.toLocaleString('pt-BR');

async function refresh() {
  const r = await fetch('/status'); const s = await r.json();
  $('#s-chars').textContent = fmt(s.corpus.chars);
  $('#s-words').textContent = fmt(s.corpus.words);
  $('#s-vocab').textContent = fmt(s.corpus.vocab);
  $('#s-mats').textContent  = s.materials.length;
  $('#matlist').innerHTML = s.materials.length ? s.materials.map(m =>
    `<div class="mat"><span>📄 ${m.name} <span style="color:#5b6b7a">· ${fmt(m.size)} B</span></span>
     <button onclick="removeMat('${m.name.replace(/'/g,"\\'")}')">✕</button></div>`).join('') :
    '<div style="color:#7d8ea0;font-size:13px">Nenhum material ainda.</div>';
  const b = $('#train-badge');
  b.textContent = s.training ? 'treinando…' : 'ocioso';
  b.className = 'badge' + (s.training ? ' live' : '');
  $('#btn-train').disabled = s.training;
  $('#btn-stop').disabled = !s.training;
  $('#btn-gen').disabled = !s.has_checkpoint;
  $('#gen-hint').style.display = s.has_checkpoint ? 'none' : 'block';
  if (s.training) { const lr = await fetch('/log'); const t = await lr.text();
    const el = $('#log'); el.textContent = t || '(iniciando…)'; el.scrollTop = el.scrollHeight; }
}

function flash(id, msg, ok) { const e = $(id); e.textContent = msg; e.className = 'msg ' + (ok?'ok':'err'); }

async function uploadFiles(files) {
  for (const f of files) {
    flash('#add-msg', `Enviando ${f.name}…`, true);
    const r = await fetch('/upload?name=' + encodeURIComponent(f.name),
                          { method:'POST', body: f });
    const j = await r.json(); flash('#add-msg', j.msg, j.ok); await refresh();
  }
}
window.removeMat = async name => {
  const r = await fetch('/remove', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})}); const j = await r.json(); flash('#add-msg', j.msg, j.ok); refresh();
};

const drop = $('#drop'), file = $('#file');
drop.onclick = () => file.click();
file.onchange = () => uploadFiles(file.files);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); uploadFiles(e.dataTransfer.files); };

$('#addtext').onclick = async () => {
  const text = $('#paste').value;
  const r = await fetch('/add_text', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text})}); const j = await r.json();
  flash('#add-msg', j.msg, j.ok); if (j.ok) $('#paste').value=''; refresh();
};

$('#btn-train').onclick = async () => {
  const body = { preset:$('#preset').value, tokenizer:$('#tokenizer').value,
    steps:$('#steps').value, dropout:$('#dropout').value, bpe_vocab:512 };
  const r = await fetch('/train', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)}); const j = await r.json(); flash('#train-msg', j.msg, j.ok); refresh();
};
$('#btn-stop').onclick = async () => {
  const r = await fetch('/stop', {method:'POST'}); const j = await r.json();
  flash('#train-msg', j.msg, j.ok); refresh();
};

$('#btn-gen').onclick = async () => {
  const btn = $('#btn-gen'); btn.disabled = true; $('#out').textContent = 'Gerando…';
  const r = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt:$('#prompt').value, max_tokens:$('#maxtok').value})});
  const j = await r.json(); $('#out').textContent = j.text || '(sem saída)'; btn.disabled = false;
};

refresh(); setInterval(refresh, 2500);
</script>
</body></html>"""


def seed_materials():
    """Na primeira execução, preserva o corpus atual como material inicial.

    Assim o usuário não perde o corpus já existente ao adicionar o primeiro
    arquivo (que dispara a reconstrução a partir de materials/).
    """
    os.makedirs(MATERIALS, exist_ok=True)
    if not list_materials() and os.path.exists(CORPUS):
        with open(CORPUS, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            with open(os.path.join(MATERIALS, "00_corpus_inicial.txt"),
                      "w", encoding="utf-8") as f:
                f.write(text)


def main():
    port = int(os.environ.get("PORT", 8000))
    seed_materials()
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  EVA · Painel de Estudo rodando em  http://localhost:{port}\n")
    print("  Abra no navegador. Ctrl+C para encerrar.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Encerrado.")


if __name__ == "__main__":
    main()
