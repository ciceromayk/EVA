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

import base64
import hmac
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
# Se EVA_PASSWORD estiver definida, o painel exige login (usuário "eva").
# Sem ela, roda aberto — apropriado só para uso 100% local (localhost).
AUTH_PASSWORD = os.environ.get("EVA_PASSWORD", "")
MATERIALS = os.path.join(BASE, "materials")
CORPUS = os.path.join(BASE, "data", "corpus.txt")
LOG = os.path.join(BASE, "train_run.log")
CKPT = os.path.join(BASE, "eva_checkpoint.pkl")
ALLOWED_EXT = (".pdf", ".txt", ".md")

sys.path.insert(0, BASE)
from tools.build_corpus import build  # noqa: E402
from tools.fetch_online_corpus import fetch_gutenberg, fetch_wikipedia  # noqa: E402

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
        resume = bool(opts.get("resume"))
        if resume and not os.path.exists(CKPT):
            return False, "Nenhum cérebro salvo para continuar. Treine do zero primeiro."
        if not resume and corpus_stats()["chars"] < 200:
            return False, "Corpus muito pequeno. Adicione material primeiro."
        cmd = [
            sys.executable, os.path.join(BASE, "train.py"),
            "--steps", str(int(opts.get("steps", 1500))),
            "--device", "gpu" if opts.get("device") == "gpu" else "cpu",
            "--log-every", "25",
        ]
        if resume:
            cmd += ["--resume"]
        else:
            cmd += [
                "--preset", str(opts.get("preset", "medium")),
                "--tokenizer", str(opts.get("tokenizer", "char")),
                "--dropout", str(float(opts.get("dropout", 0.1))),
            ]
            if opts.get("tokenizer") == "bpe":
                cmd += ["--bpe-vocab", str(int(opts.get("bpe_vocab", 512)))]
        logfile = open(LOG, "w")
        _train_proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                       cwd=BASE, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return True, "Continuando treino do cérebro salvo…" if resume else "Treino iniciado."


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

    def _send_file(self, path, filename):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        """Confere usuário/senha (HTTP Basic Auth). Sem EVA_PASSWORD, libera tudo."""
        if not AUTH_PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            user, _, pwd = decoded.partition(":")
        except Exception:
            return False
        return user == "eva" and hmac.compare_digest(pwd, AUTH_PASSWORD)

    def _require_auth(self) -> bool:
        """Se não autorizado, envia o desafio de login e devolve False."""
        if self._authorized():
            return True
        body = b"Autenticacao necessaria."
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="EVA"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    # ------------------------------------------------------------------
    def do_GET(self):
        if not self._require_auth():
            return
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
        elif path == "/download_model":
            if os.path.exists(CKPT):
                self._send_file(CKPT, "eva_cerebro.pkl")
            else:
                self._send(404, "sem modelo treinado", "text/plain; charset=utf-8")
        elif path == "/download_corpus":
            if os.path.exists(CORPUS):
                self._send_file(CORPUS, "eva_corpus.txt")
            else:
                self._send(404, "sem corpus", "text/plain; charset=utf-8")
        else:
            self._send(404, "não encontrado", "text/plain; charset=utf-8")

    def do_POST(self):
        if not self._require_auth():
            return
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

            elif path == "/upload_model":
                if training_active():
                    return self._json({"ok": False, "msg": "Pare o treino antes de restaurar."}, 400)
                body = self._body()
                if len(body) < 10:
                    return self._json({"ok": False, "msg": "Arquivo vazio."}, 400)
                with open(CKPT, "wb") as f:
                    f.write(body)
                self._json({"ok": True, "msg": "Cérebro restaurado. Já dá para gerar texto."})

            elif path == "/fetch_online":
                data = json.loads(self._body() or b"{}")
                source = data.get("source")
                query = (data.get("query") or "").strip()
                lang = (data.get("lang") or "pt").strip() or "pt"
                if not query:
                    return self._json({"ok": False, "msg": "Digite um termo de busca."}, 400)
                os.makedirs(MATERIALS, exist_ok=True)
                try:
                    if source == "wikipedia":
                        titles = [t.strip() for t in query.split(",") if t.strip()]
                        saved = fetch_wikipedia(titles, lang=lang, out_dir=MATERIALS)
                    elif source == "gutenberg":
                        saved = fetch_gutenberg(query, lang=lang, max_books=5, out_dir=MATERIALS)
                    else:
                        return self._json({"ok": False, "msg": "Fonte desconhecida."}, 400)
                except Exception as exc:
                    return self._json({"ok": False, "msg": f"Falha na busca online: {exc}"}, 502)
                if not saved:
                    return self._json({"ok": False, "msg": "Nada encontrado para essa busca."})
                stats = rebuild_corpus()
                nomes = ", ".join(os.path.basename(p) for p in saved)
                self._json({"ok": True, "msg": f"Adicionado(s): {nomes}", "corpus": stats})

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
<title>EVA · Núcleo Neural</title>
<style>
:root{
  --bg:#05060c; --cyan:#22d3ee; --mag:#e152ff; --vio:#7c5cff; --lime:#7cf67a;
  --ink:#eaf2ff; --mut:#93a4c4; --glass:rgba(20,26,44,.55); --line:rgba(124,160,255,.18);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  font-family:'Segoe UI',system-ui,-apple-system,Roboto,sans-serif; color:var(--ink);
  background:var(--bg); overflow-x:hidden; position:relative; min-height:100%;
}
/* fundo aurora animado */
body::before{
  content:""; position:fixed; inset:-30%; z-index:-2;
  background:
    radial-gradient(40% 40% at 20% 20%, rgba(34,211,238,.18), transparent 60%),
    radial-gradient(45% 45% at 82% 25%, rgba(225,82,255,.16), transparent 60%),
    radial-gradient(50% 50% at 50% 90%, rgba(124,92,255,.18), transparent 60%);
  filter:blur(30px); animation:drift 22s ease-in-out infinite alternate;
}
body::after{ /* grade futurista */
  content:""; position:fixed; inset:0; z-index:-1; opacity:.35;
  background-image:linear-gradient(var(--line) 1px,transparent 1px),
                   linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:44px 44px; mask-image:radial-gradient(circle at 50% 30%,#000,transparent 80%);
}
@keyframes drift{0%{transform:translate(-4%,-2%) scale(1)}100%{transform:translate(4%,3%) scale(1.1)}}

header{display:flex;align-items:center;gap:18px;padding:26px 30px;position:relative}
.orb{width:60px;height:60px;border-radius:50%;position:relative;flex:0 0 auto;
  background:radial-gradient(circle at 35% 30%,#bffcff,var(--cyan) 40%,var(--vio) 85%);
  box-shadow:0 0 30px rgba(34,211,238,.55),0 0 60px rgba(124,92,255,.35),inset 0 0 18px rgba(255,255,255,.5);
  animation:breathe 3.4s ease-in-out infinite}
.orb::before{content:"";position:absolute;inset:-8px;border-radius:50%;
  background:conic-gradient(from 0deg,transparent,var(--cyan),transparent 30%,var(--mag),transparent 70%);
  opacity:.0;transition:opacity .4s;animation:spin 3.5s linear infinite}
.orb.busy{animation:breathe 1s ease-in-out infinite}
.orb.busy::before{opacity:.9}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.09)}}
@keyframes spin{to{transform:rotate(360deg)}}
.brand h1{margin:0;font-size:22px;font-weight:700;letter-spacing:.5px}
.brand .sub{font-size:13px;color:var(--mut)}
.brand b{background:linear-gradient(90deg,var(--cyan),var(--mag));-webkit-background-clip:text;background-clip:text;color:transparent}

main{max-width:960px;margin:0 auto;padding:8px 22px 60px;display:grid;gap:20px}
.card{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:22px;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.05);
  position:relative;overflow:hidden}
.card::after{content:"";position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),var(--mag),transparent);opacity:.5}
.card h2{margin:0 0 16px;font-size:14px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--mut);display:flex;align-items:center;gap:9px}
.card h2 .ic{font-size:17px;filter:drop-shadow(0 0 6px var(--cyan))}

.stats{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}
.stat{flex:1;min-width:120px;background:rgba(8,12,24,.6);border:1px solid var(--line);
  border-radius:14px;padding:14px 16px;position:relative}
.stat b{display:block;font-size:26px;font-weight:800;font-variant-numeric:tabular-nums;
  background:linear-gradient(90deg,var(--cyan),var(--vio));-webkit-background-clip:text;background-clip:text;color:transparent}
.stat span{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--mut)}

.mat{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;
  background:rgba(8,12,24,.5);border:1px solid var(--line);border-radius:12px;margin-bottom:8px;font-size:14px}
.mat button{background:none;border:none;color:#ff7b9c;cursor:pointer;font-size:16px;transition:.2s}
.mat button:hover{transform:scale(1.3);filter:drop-shadow(0 0 6px #ff7b9c)}

label{font-size:12px;letter-spacing:.5px;color:var(--mut);display:block;margin:12px 0 5px}
input,select,textarea{width:100%;background:rgba(6,10,20,.7);border:1px solid var(--line);color:var(--ink);
  border-radius:11px;padding:11px 13px;font-size:14px;font-family:inherit;transition:.2s;outline:none}
input:focus,select:focus,textarea:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(34,211,238,.15)}
textarea{resize:vertical}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:130px}

button.primary{position:relative;background:linear-gradient(90deg,var(--cyan),var(--vio));color:#03040a;
  border:none;border-radius:12px;padding:12px 22px;font-size:14px;font-weight:700;cursor:pointer;
  letter-spacing:.4px;transition:.2s;box-shadow:0 6px 20px rgba(34,211,238,.3)}
button.primary:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(124,92,255,.5)}
button.primary:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
button.ghost{background:rgba(8,12,24,.6);color:var(--ink);border:1px solid var(--line);border-radius:12px;
  padding:11px 18px;cursor:pointer;font-size:14px;transition:.2s}
button.ghost:hover{border-color:var(--mag);color:#fff}
button.ghost:disabled{opacity:.4;cursor:not-allowed}

.drop{border:1.5px dashed rgba(124,160,255,.35);border-radius:16px;padding:30px;text-align:center;
  color:var(--mut);cursor:pointer;transition:.25s;background:rgba(8,12,24,.35)}
.drop:hover{border-color:var(--cyan);color:var(--ink)}
.drop.over{border-color:var(--cyan);background:rgba(34,211,238,.08);color:#cfe0ff;transform:scale(1.01)}
.drop b{color:var(--cyan)}

pre{background:#04060d;border:1px solid var(--line);border-radius:12px;padding:15px;overflow:auto;
  max-height:300px;font-size:12.5px;line-height:1.55;white-space:pre-wrap;color:#9be89b;
  font-family:'Cascadia Code',Consolas,monospace;text-shadow:0 0 8px rgba(124,246,122,.3)}

.badge{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;padding:4px 11px;border-radius:20px;
  background:rgba(124,160,255,.12);color:var(--mut);border:1px solid var(--line)}
.badge.live{background:rgba(124,246,122,.12);color:var(--lime);border-color:rgba(124,246,122,.4);
  animation:blink 1.4s ease-in-out infinite}
@keyframes blink{50%{opacity:.55}}

.prog{height:10px;border-radius:20px;background:rgba(8,12,24,.7);border:1px solid var(--line);
  overflow:hidden;margin:6px 0 4px;display:none}
.prog .fill{height:100%;width:0;border-radius:20px;
  background:linear-gradient(90deg,var(--cyan),var(--mag));box-shadow:0 0 14px var(--cyan);transition:width .5s}
.proglabel{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}

.out{background:#04060d;border:1px solid var(--line);border-radius:14px;padding:18px;min-height:64px;
  font-size:15px;line-height:1.7;white-space:pre-wrap;color:#dbe7ff}
.out.think{color:var(--mut);font-style:italic}
.msg{font-size:13px;margin-top:11px;min-height:18px}.ok{color:var(--lime)}.err{color:#ff7b9c}
.hint{font-size:12px;color:var(--mut);margin-top:7px}
.checkrow{display:flex;align-items:center;gap:9px;margin-bottom:14px;cursor:pointer;user-select:none}
.checkrow input{width:auto;accent-color:var(--cyan);cursor:pointer}
.checkrow span{font-size:14px}
.archonly.dim{opacity:.35;pointer-events:none}
.foot{text-align:center;color:var(--mut);font-size:12px;padding:10px}
</style></head>
<body>
<header>
  <div class="orb" id="orb"></div>
  <div class="brand"><h1><b>EVA</b> · Núcleo Neural</h1>
    <div class="sub">alimente · treine · converse — sua IA feita do zero</div></div>
</header>
<main>

  <div class="card">
    <h2><span class="ic">🧬</span> Memória da EVA</h2>
    <div class="stats">
      <div class="stat"><b id="s-chars">–</b><span>caracteres</span></div>
      <div class="stat"><b id="s-words">–</b><span>palavras</span></div>
      <div class="stat"><b id="s-vocab">–</b><span>vocabulário</span></div>
      <div class="stat"><b id="s-mats">–</b><span>materiais</span></div>
    </div>
    <div id="matlist"></div>
  </div>

  <div class="card">
    <h2><span class="ic">📡</span> Alimentar conhecimento</h2>
    <div class="drop" id="drop">⬆ Solte PDFs/TXT aqui ou <b>clique para escolher</b>
      <input type="file" id="file" multiple accept=".pdf,.txt,.md" style="display:none"></div>
    <label>…ou injete um texto direto</label>
    <textarea id="paste" rows="4" placeholder="Cole qualquer texto para a EVA absorver…"></textarea>
    <div style="margin-top:11px"><button class="ghost" id="addtext">+ Injetar texto</button></div>
    <div class="msg" id="add-msg"></div>
  </div>

  <div class="card">
    <h2><span class="ic">🌐</span> Buscar conhecimento online</h2>
    <div class="row">
      <div style="flex:2"><label>Fonte</label><select id="onlinesrc">
        <option value="wikipedia">Wikipédia (artigos por título)</option>
        <option value="gutenberg">Project Gutenberg (livros de domínio público)</option></select></div>
      <div><label>Idioma</label><input id="onlinelang" value="pt" maxlength="5"></div>
    </div>
    <label>Termo de busca</label>
    <input id="onlinequery" placeholder="Wikipédia: 'Inteligência artificial,Redes neurais' · Gutenberg: 'Machado de Assis'">
    <div class="hint">Wikipédia aceita vários títulos separados por vírgula. Gutenberg busca por
      autor/título e traz até 5 livros.</div>
    <div style="margin-top:11px"><button class="ghost" id="btn-online">🌐 Buscar e adicionar</button></div>
    <div class="msg" id="online-msg"></div>
  </div>

  <div class="card">
    <h2><span class="ic">⚡</span> Treinar a mente <span class="badge" id="train-badge">em repouso</span></h2>
    <label class="checkrow" for="resume"><input type="checkbox" id="resume">
      <span>🔄 Continuar do cérebro salvo (em vez de começar um modelo novo)</span></label>
    <div class="row">
      <div class="archonly"><label>Tamanho do cérebro</label><select id="preset">
        <option value="nano">nano · relâmpago</option>
        <option value="small" selected>small · rápido</option>
        <option value="medium">medium · esperto</option>
        <option value="large">large · lento</option></select></div>
      <div class="archonly"><label>Percepção</label><select id="tokenizer">
        <option value="char">char · letra a letra</option>
        <option value="bpe">bpe · subpalavras</option></select></div>
      <div><label>Ciclos (passos)</label><input type="number" id="steps" value="1000" min="100" step="100"></div>
      <div class="archonly"><label>Dropout</label><input type="number" id="dropout" value="0.1" min="0" max="0.9" step="0.05"></div>
      <div><label>Processador</label><select id="device">
        <option value="cpu">CPU</option>
        <option value="gpu">GPU · CUDA</option></select></div>
    </div>
    <div class="hint" id="resume-hint" style="display:none">Continuando: usa a arquitetura, tokenizer e
      dropout do cérebro já salvo. Só "Ciclos" e "Processador" continuam valendo (acima).</div>
    <div style="margin-top:18px;display:flex;gap:11px">
      <button class="primary" id="btn-train">⚡ Iniciar treino</button>
      <button class="ghost" id="btn-stop">■ Parar</button>
    </div>
    <div class="msg" id="train-msg"></div>
    <div class="prog" id="prog"><div class="fill" id="progfill"></div></div>
    <div class="proglabel" id="proglabel"></div>
    <label style="margin-top:12px">Fluxo neural</label>
    <pre id="log">&gt; aguardando ativação…</pre>
  </div>

  <div class="card">
    <h2><span class="ic">💾</span> Salvar / restaurar cérebro</h2>
    <div class="hint" style="margin:0 0 12px">Baixe o modelo treinado para o seu dispositivo e restaure depois — em
      qualquer aparelho. Assim o progresso não se perde mesmo se o servidor reiniciar.</div>
    <div style="display:flex;gap:11px;flex-wrap:wrap">
      <button class="ghost" id="btn-dl">⬇ Baixar cérebro</button>
      <button class="ghost" id="btn-ul">⬆ Restaurar cérebro</button>
      <input type="file" id="modelfile" accept=".pkl" style="display:none">
    </div>
    <div class="msg" id="model-msg"></div>
  </div>

  <div class="card">
    <h2><span class="ic">💬</span> Conversar com a EVA</h2>
    <label>Semente do pensamento (prompt)</label>
    <input id="prompt" value="A arte da guerra" placeholder="Comece uma frase…">
    <div class="row" style="margin-top:11px;align-items:end">
      <div><label>Extensão (tokens)</label><input type="number" id="maxtok" value="250" min="20" max="800" step="10"></div>
      <div style="flex:0"><button class="primary" id="btn-gen">✨ Gerar</button></div>
    </div>
    <div class="hint" id="gen-hint">Requer um treino concluído.</div>
    <div class="out" id="out" style="margin-top:13px"></div>
  </div>

  <div class="foot">EVA — Transformer construído do zero em NumPy · roda 100% local</div>
</main>
<script>
const $=s=>document.querySelector(s);
const fmt=n=>n.toLocaleString('pt-BR');

async function refresh(){
  const s=await(await fetch('/status')).json();
  $('#s-chars').textContent=fmt(s.corpus.chars);
  $('#s-words').textContent=fmt(s.corpus.words);
  $('#s-vocab').textContent=fmt(s.corpus.vocab);
  $('#s-mats').textContent=s.materials.length;
  $('#matlist').innerHTML=s.materials.length?s.materials.map(m=>
    `<div class="mat"><span>🧩 ${m.name} <span style="color:#5b6b7a">· ${fmt(m.size)} B</span></span>
     <button onclick="removeMat('${m.name.replace(/'/g,"\\'")}')">✕</button></div>`).join(''):
    '<div style="color:var(--mut);font-size:13px">Nenhum material ainda — alimente a EVA abaixo.</div>';
  const b=$('#train-badge'),orb=$('#orb');
  b.textContent=s.training?'treinando':'em repouso';
  b.className='badge'+(s.training?' live':'');
  orb.className='orb'+(s.training?' busy':'');
  $('#btn-train').disabled=s.training;
  $('#btn-stop').disabled=!s.training;
  $('#btn-gen').disabled=!s.has_checkpoint;
  $('#btn-dl').disabled=!s.has_checkpoint;
  $('#gen-hint').style.display=s.has_checkpoint?'none':'block';
  $('#resume').disabled=!s.has_checkpoint;
  if(!s.has_checkpoint)$('#resume').checked=false;
  updateResumeUI();
  if(s.training){
    const t=await(await fetch('/log')).text();
    const el=$('#log');el.textContent=t||'> iniciando…';el.scrollTop=el.scrollHeight;
    const ms=[...t.matchAll(/passo\s+(\d+)\/(\d+)\s+\|\s+treino\s+([\d.]+)\s+\|\s+val\s+([\d.]+)/g)];
    const prog=$('#prog');
    if(ms.length){const m=ms[ms.length-1],cur=+m[1],tot=+m[2];
      prog.style.display='block';$('#progfill').style.width=(100*cur/tot)+'%';
      $('#proglabel').textContent=`passo ${cur}/${tot} · treino ${m[3]} · val ${m[4]}`;}
    else{prog.style.display='none';$('#proglabel').textContent='';}
  }else{$('#prog').style.display='none';$('#proglabel').textContent='';}
}
function flash(id,msg,ok){const e=$(id);e.textContent=msg;e.className='msg '+(ok?'ok':'err');}

async function uploadFiles(files){
  for(const f of files){flash('#add-msg',`📡 Enviando ${f.name}…`,true);
    const j=await(await fetch('/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f})).json();
    flash('#add-msg',j.msg,j.ok);await refresh();}
}
window.removeMat=async name=>{
  const j=await(await fetch('/remove',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})})).json();flash('#add-msg',j.msg,j.ok);refresh();};

const drop=$('#drop'),file=$('#file');
drop.onclick=()=>file.click();
file.onchange=()=>uploadFiles(file.files);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');uploadFiles(e.dataTransfer.files)};

$('#addtext').onclick=async()=>{
  const j=await(await fetch('/add_text',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:$('#paste').value})})).json();
  flash('#add-msg',j.msg,j.ok);if(j.ok)$('#paste').value='';refresh();};

$('#btn-online').onclick=async()=>{
  const btn=$('#btn-online');btn.disabled=true;
  flash('#online-msg','🌐 Buscando… pode levar alguns segundos',true);
  const body={source:$('#onlinesrc').value,query:$('#onlinequery').value,lang:$('#onlinelang').value};
  const j=await(await fetch('/fetch_online',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)})).json();
  flash('#online-msg',j.msg,j.ok);btn.disabled=false;if(j.ok)$('#onlinequery').value='';refresh();};

function updateResumeUI(){
  const resuming=$('#resume').checked;
  document.querySelectorAll('.archonly').forEach(el=>el.classList.toggle('dim',resuming));
  document.querySelectorAll('.archonly select,.archonly input').forEach(el=>el.disabled=resuming);
  $('#resume-hint').style.display=resuming?'block':'none';
}
$('#resume').onchange=updateResumeUI;

$('#btn-train').onclick=async()=>{
  const resume=$('#resume').checked;
  const body={preset:$('#preset').value,tokenizer:$('#tokenizer').value,
    steps:$('#steps').value,dropout:$('#dropout').value,device:$('#device').value,
    bpe_vocab:512,resume};
  const j=await(await fetch('/train',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)})).json();flash('#train-msg',j.msg,j.ok);refresh();};
$('#btn-stop').onclick=async()=>{
  const j=await(await fetch('/stop',{method:'POST'})).json();flash('#train-msg',j.msg,j.ok);refresh();};

$('#btn-gen').onclick=async()=>{
  const btn=$('#btn-gen'),out=$('#out');btn.disabled=true;
  out.className='out think';out.textContent='✨ pensando…';
  const j=await(await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:$('#prompt').value,max_tokens:$('#maxtok').value})})).json();
  out.className='out';out.textContent=j.text||'(sem saída)';btn.disabled=false;};

$('#btn-dl').onclick=()=>{window.location='/download_model'};
$('#btn-ul').onclick=()=>$('#modelfile').click();
$('#modelfile').onchange=async()=>{
  const f=$('#modelfile').files[0];if(!f)return;
  flash('#model-msg','⬆ Restaurando…',true);
  const j=await(await fetch('/upload_model',{method:'POST',body:f})).json();
  flash('#model-msg',j.msg,j.ok);refresh();};

refresh();setInterval(refresh,2000);
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
    if AUTH_PASSWORD:
        print("  🔒 Login exigido (usuário: eva). Seguro para acessar de fora.\n")
    else:
        print("  ⚠ Sem senha (EVA_PASSWORD não definida). Só acesse por")
        print("    localhost — NÃO exponha assim para fora desta máquina.\n")
    print("  Abra no navegador. Ctrl+C para encerrar.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Encerrado.")


if __name__ == "__main__":
    main()
