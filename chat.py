"""EVA · Sala de Conversa — interface web dedicada a USAR o modelo.

Diferente do painel (`app.py`), que relança o `train.py` a cada geração,
aqui o cérebro é carregado UMA vez na memória e os tokens são transmitidos
ao vivo para o navegador, um a um, conforme saem do modelo (streaming).

Uso:
    python chat.py                 # http://localhost:8001
    EVA_PASSWORD=segredo python chat.py   # exige login (usuário: eva)

A EVA é um modelo de CONTINUAÇÃO (não é instruída a dialogar): você
escreve um começo de texto e ela continua no seu estilo do corpus.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# O dispositivo precisa ser escolhido antes de importar `eva` (ver train.py).
import numpy as np  # noqa: E402

from train import CKPT_PATH, load_checkpoint  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, CKPT_PATH)
AUTH_PASSWORD = os.environ.get("EVA_PASSWORD", "")

# ----------------------------------------------------------------------
# Cérebro em memória (carregado uma vez; recarrega se o checkpoint mudar)
# ----------------------------------------------------------------------
_brain = {"mtime": None, "model": None, "tokenizer": None, "error": None}
_brain_lock = threading.Lock()
GEN_LOCK = threading.Lock()  # o modelo gera uma resposta por vez


def get_brain():
    """Devolve (model, tokenizer) ou (None, None) se não há checkpoint."""
    if not os.path.exists(CKPT):
        return None, None
    mtime = os.path.getmtime(CKPT)
    with _brain_lock:
        if _brain["mtime"] != mtime:
            try:
                model, tokenizer = load_checkpoint()
                _brain.update(mtime=mtime, model=model, tokenizer=tokenizer, error=None)
                print(f"  Cérebro carregado: {model.num_params():,} parâmetros")
            except SystemExit as exc:  # checkpoint da era GPT (incompatível)
                _brain.update(mtime=mtime, model=None, tokenizer=None, error=str(exc))
        return _brain["model"], _brain["tokenizer"]


def brain_info() -> dict:
    model, tokenizer = get_brain()
    if model is None:
        return {"ready": False,
                "error": _brain["error"] or "Nenhum cérebro treinado ainda. "
                "Treine a EVA no painel (iniciar.bat / app.py) primeiro."}
    cfg = model.config
    kind = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    return {"ready": True, "params": model.num_params(), "n_layer": cfg.n_layer,
            "n_head": cfg.n_head, "n_embd": cfg.n_embd, "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size, "tokenizer": "bpe" if kind == "bpe" else "char"}


def stream_reply(prompt: str, max_tokens: int, temperature: float, top_k: int):
    """Generator de PEDAÇOS DE TEXTO novos, prontos para enviar ao navegador.

    Decodifica o histórico inteiro a cada token e emite só o sufixo novo —
    assim tokens BPE que formam um caractere multi-byte só aparecem quando
    o texto está estável (evita mandar caracteres pela metade).
    """
    model, tokenizer = get_brain()
    ids = tokenizer.encode(prompt) or [0]
    new_ids: list[int] = []
    sent = ""
    for tid in model.stream(np.array([ids], dtype=np.int64), max_tokens,
                            temperature=max(temperature, 1e-3),
                            top_k=top_k if top_k > 0 else None):
        new_ids.append(tid)
        text = tokenizer.decode(new_ids)
        if text.startswith(sent) and len(text) > len(sent):
            chunk, sent = text[len(sent):], text
            yield chunk


# ----------------------------------------------------------------------
# Servidor HTTP
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
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

    def _authorized(self) -> bool:
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

    def do_GET(self):
        if not self._require_auth():
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/info":
            self._json(brain_info())
        else:
            self._send(404, "não encontrado", "text/plain; charset=utf-8")

    def do_POST(self):
        if not self._require_auth():
            return
        path = urllib.parse.urlparse(self.path).path
        if path != "/generate":
            self._send(404, "não encontrado", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "JSON inválido"}, 400)
            return

        if not brain_info()["ready"]:
            self._json({"error": brain_info()["error"]}, 503)
            return
        if not GEN_LOCK.acquire(blocking=False):
            self._json({"error": "A EVA ainda está escrevendo a resposta "
                        "anterior. Aguarde um instante."}, 409)
            return
        try:
            prompt = str(data.get("prompt", ""))[:4000]
            max_tokens = min(max(int(data.get("max_tokens", 300)), 1), 2000)
            temperature = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 10))

            # Resposta SEM Content-Length: em HTTP/1.0 o fim da conexão marca
            # o fim do corpo, e o navegador lê os pedaços conforme chegam.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for chunk in stream_reply(prompt, max_tokens, temperature, top_k):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # o usuário apertou "Parar" / fechou a aba: interrompe
        finally:
            GEN_LOCK.release()


# ----------------------------------------------------------------------
# Página (HTML + CSS + JS, tudo embutido — sem dependências externas)
# ----------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVA · Sala de Conversa</title>
<style>
:root{
  --bg:#0d0a06; --panel:rgba(24,19,12,.72); --line:#2e2617; --line-soft:#221c10;
  --amber:#ffb454; --amber-hi:#ffd9a0; --amber-dim:#8a6a3a;
  --ink:#eae3d3; --ink-dim:#9a917d; --red:#ff6b6b;
  --mono:"Cascadia Mono","JetBrains Mono",Consolas,ui-monospace,"SF Mono",Menlo,monospace;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg); color:var(--ink); font-family:var(--serif);
  background-image:
    radial-gradient(ellipse 90% 70% at 50% -10%, rgba(255,180,84,.08), transparent 60%),
    radial-gradient(ellipse 60% 50% at 85% 110%, rgba(255,180,84,.05), transparent 60%);
  overflow:hidden;
}
/* scanlines + vinheta de tubo (CRT) sobre tudo, sem capturar cliques */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9;
  background:repeating-linear-gradient(0deg, rgba(0,0,0,.16) 0 1px, transparent 1px 3px);}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:9;
  background:radial-gradient(ellipse 120% 100% at 50% 45%, transparent 55%, rgba(0,0,0,.5));}

.app{display:grid;grid-template-columns:290px 1fr;grid-template-rows:auto 1fr;height:100%;
  max-width:1200px;margin:0 auto;padding:20px 22px;gap:16px}
header{grid-column:1/-1;display:flex;align-items:baseline;gap:14px;
  border-bottom:1px solid var(--line);padding-bottom:14px;
  animation:reveal .7s cubic-bezier(.2,.7,.2,1) both}
header h1{font-size:26px;font-weight:400;letter-spacing:.5px}
header h1 b{color:var(--amber);font-weight:700;text-shadow:0 0 18px rgba(255,180,84,.55)}
header .sub{font-family:var(--mono);font-size:11.5px;color:var(--ink-dim);letter-spacing:.14em;
  text-transform:uppercase}
header .cursor{display:inline-block;width:9px;height:17px;background:var(--amber);
  vertical-align:-2px;margin-left:4px;animation:blink 1.1s steps(1) infinite;
  box-shadow:0 0 10px rgba(255,180,84,.8)}
@keyframes blink{50%{opacity:0}}
@keyframes reveal{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* ---------- ficha do cérebro ---------- */
aside{display:flex;flex-direction:column;gap:14px;overflow-y:auto;padding-right:2px;
  animation:reveal .7s .12s cubic-bezier(.2,.7,.2,1) both}
.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px 16px 14px;
  backdrop-filter:blur(6px)}
.card h2{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--amber-dim);margin-bottom:12px}
.spec{display:flex;justify-content:space-between;font-family:var(--mono);font-size:12.5px;
  padding:4px 0;border-bottom:1px dashed var(--line-soft)}
.spec:last-child{border-bottom:none}
.spec .k{color:var(--ink-dim)} .spec .v{color:var(--amber-hi)}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.badge{font-family:var(--mono);font-size:10px;letter-spacing:.08em;color:var(--amber);
  border:1px solid var(--amber-dim);border-radius:2px;padding:3px 7px;
  background:rgba(255,180,84,.06)}
.hint{font-size:13.5px;line-height:1.55;color:var(--ink-dim);font-style:italic}
.hint b{color:var(--ink);font-style:normal}
label{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-dim);margin:10px 0 5px}
label:first-of-type{margin-top:0}
input[type=number]{width:100%;background:rgba(10,8,4,.8);border:1px solid var(--line);
  color:var(--amber-hi);font-family:var(--mono);font-size:13px;padding:7px 9px;border-radius:3px}
input[type=number]:focus{outline:none;border-color:var(--amber-dim);
  box-shadow:0 0 0 3px rgba(255,180,84,.12)}
input[type=range]{width:100%;accent-color:var(--amber)}
.rangerow{display:flex;align-items:center;gap:10px}
.rangerow output{font-family:var(--mono);font-size:12.5px;color:var(--amber-hi);min-width:34px;
  text-align:right}

/* ---------- manuscrito ---------- */
main{display:flex;flex-direction:column;min-height:0;
  animation:reveal .7s .22s cubic-bezier(.2,.7,.2,1) both}
#scroll{flex:1;overflow-y:auto;border:1px solid var(--line);border-radius:4px;
  background:var(--panel);backdrop-filter:blur(6px);padding:26px 30px;
  font-family:var(--mono);font-size:14.5px;line-height:1.85;scroll-behavior:smooth}
#scroll::-webkit-scrollbar{width:10px}
#scroll::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
.entry{margin-bottom:30px;animation:reveal .4s both}
.entry .stamp{font-size:10px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--amber-dim);margin-bottom:8px;user-select:none}
.entry .text{white-space:pre-wrap;word-break:break-word}
.entry .you{color:var(--ink)}
.entry .eva{color:var(--amber-hi);text-shadow:0 0 14px rgba(255,180,84,.28)}
.entry.live .eva::after{content:"▮";color:var(--amber);animation:blink .9s steps(1) infinite;
  text-shadow:0 0 12px rgba(255,180,84,.9)}
.empty{color:var(--ink-dim);font-family:var(--serif);font-style:italic;font-size:15.5px;
  line-height:1.7;max-width:520px}
.empty b{color:var(--amber);font-style:normal}
.error{color:var(--red);font-size:13px}

/* ---------- entrada ---------- */
.compose{display:flex;gap:10px;margin-top:14px;align-items:flex-end}
.compose textarea{flex:1;background:rgba(10,8,4,.85);border:1px solid var(--line);
  color:var(--ink);font-family:var(--mono);font-size:14px;line-height:1.6;
  padding:12px 14px;border-radius:3px;resize:none;height:76px}
.compose textarea:focus{outline:none;border-color:var(--amber-dim);
  box-shadow:0 0 0 3px rgba(255,180,84,.12)}
.compose textarea::placeholder{color:#5c5342;font-style:italic}
button{font-family:var(--mono);cursor:pointer;border-radius:3px;font-size:13px;
  letter-spacing:.06em;transition:.18s}
#send{background:var(--amber);color:#140d02;border:1px solid var(--amber);
  padding:12px 22px;font-weight:700;text-shadow:none;
  box-shadow:0 0 22px rgba(255,180,84,.35)}
#send:hover{background:var(--amber-hi);box-shadow:0 0 30px rgba(255,180,84,.55)}
#send:disabled{opacity:.35;cursor:not-allowed;box-shadow:none}
#stop{background:transparent;color:var(--red);border:1px solid rgba(255,107,107,.5);
  padding:12px 16px;display:none}
#stop:hover{background:rgba(255,107,107,.1)}
.keys{font-family:var(--mono);font-size:10.5px;color:#5c5342;margin-top:7px;letter-spacing:.06em}

@media (max-width:900px){
  body{overflow:auto}
  .app{grid-template-columns:1fr;height:auto}
  aside{order:2} main{order:1;min-height:70vh}
}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1><b>EVA</b> · Sala de Conversa<span class="cursor"></span></h1>
    <span class="sub">arquitetura llama · numpy puro</span>
  </header>

  <aside>
    <div class="card" id="ficha">
      <h2>Ficha do cérebro</h2>
      <div id="specs"><div class="spec"><span class="k">estado</span><span class="v">consultando…</span></div></div>
      <div class="badges"><span class="badge">RoPE</span><span class="badge">RMSNorm</span>
        <span class="badge">SwiGLU</span><span class="badge">sem bias</span></div>
    </div>
    <div class="card">
      <h2>Regulagem</h2>
      <label for="maxtok">Tokens a gerar</label>
      <input type="number" id="maxtok" value="300" min="1" max="2000">
      <label for="temp">Temperatura <i style="text-transform:none;letter-spacing:0">(ousadia)</i></label>
      <div class="rangerow"><input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.8">
        <output id="tempval">0.80</output></div>
      <label for="topk">Top-k <i style="text-transform:none;letter-spacing:0">(0 = livre)</i></label>
      <input type="number" id="topk" value="10" min="0" max="500">
    </div>
    <div class="card">
      <h2>Como conversar</h2>
      <p class="hint">A EVA <b>continua</b> o que você escrever — ela aprendeu
      imitando o corpus, não a obedecer instruções. Comece uma frase e deixe
      que ela a leve adiante: <b>“A arte da guerra ensina”</b>,
      <b>“Os modelos de linguagem”</b>…</p>
    </div>
  </aside>

  <main>
    <div id="scroll">
      <div class="empty" id="empty">Escreva o começo de um pensamento abaixo
      e a <b>EVA</b> o continuará aqui, palavra por palavra, conforme os
      tokens saem da rede neural — sem truques, sem API externa: só
      <b>NumPy</b> girando em sua máquina.</div>
    </div>
    <div class="compose">
      <textarea id="prompt" placeholder="Escreva o começo… (Enter envia, Shift+Enter quebra linha)"></textarea>
      <button id="send">Continuar ✒</button>
      <button id="stop">■ Parar</button>
    </div>
    <div class="keys">ENTER envia · SHIFT+ENTER quebra linha · a continuação surge token a token</div>
  </main>
</div>

<script>
const $=id=>document.getElementById(id);
let controller=null;

const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

async function loadInfo(){
  try{
    const i=await(await fetch('/info')).json();
    if(!i.ready){
      $('specs').innerHTML=`<div class="error">${esc(i.error)}</div>`;
      $('send').disabled=true; return;
    }
    $('send').disabled=false;
    const fmt=n=>n.toLocaleString('pt-BR');
    $('specs').innerHTML=
      `<div class="spec"><span class="k">parâmetros</span><span class="v">${fmt(i.params)}</span></div>`+
      `<div class="spec"><span class="k">camadas</span><span class="v">${i.n_layer}</span></div>`+
      `<div class="spec"><span class="k">cabeças</span><span class="v">${i.n_head}</span></div>`+
      `<div class="spec"><span class="k">dimensão</span><span class="v">${i.n_embd}</span></div>`+
      `<div class="spec"><span class="k">contexto</span><span class="v">${i.block_size} tokens</span></div>`+
      `<div class="spec"><span class="k">vocabulário</span><span class="v">${fmt(i.vocab_size)} (${i.tokenizer})</span></div>`;
  }catch(e){
    $('specs').innerHTML='<div class="error">painel fora do ar?</div>';
  }
}

function setBusy(b){
  $('send').style.display=b?'none':'inline-block';
  $('stop').style.display=b?'inline-block':'none';
  $('prompt').disabled=b;
}

async function send(){
  const prompt=$('prompt').value;
  if(!prompt.trim()||controller)return;
  $('empty')?.remove();
  const entry=document.createElement('div');
  entry.className='entry live';
  entry.innerHTML=`<div class="stamp">você começou</div>`+
    `<div class="text"><span class="you">${esc(prompt)}</span><span class="eva"></span></div>`;
  $('scroll').appendChild(entry);
  const evaSpan=entry.querySelector('.eva');
  const scroll=$('scroll');
  scroll.scrollTop=scroll.scrollHeight;
  $('prompt').value='';
  setBusy(true);
  controller=new AbortController();
  try{
    const res=await fetch('/generate',{method:'POST',signal:controller.signal,
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,max_tokens:+$('maxtok').value,
        temperature:+$('temp').value,top_k:+$('topk').value})});
    if(!res.ok){
      const err=await res.json().catch(()=>({error:'erro '+res.status}));
      evaSpan.innerHTML=` <span class="error">${esc(err.error||'falhou')}</span>`;
    }else{
      const reader=res.body.getReader(), dec=new TextDecoder();
      for(;;){
        const {done,value}=await reader.read();
        if(done)break;
        evaSpan.textContent+=dec.decode(value,{stream:true});
        const nearBottom=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<160;
        if(nearBottom)scroll.scrollTop=scroll.scrollHeight;
      }
    }
  }catch(e){ /* abortado pelo botão Parar */ }
  entry.classList.remove('live');
  controller=null;
  setBusy(false);
  $('prompt').focus();
}

$('send').onclick=send;
$('stop').onclick=()=>controller&&controller.abort();
$('prompt').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
$('temp').oninput=()=>$('tempval').value=(+$('temp').value).toFixed(2);
loadInfo();
setInterval(loadInfo,15000);
</script>
</body>
</html>
"""


def main():
    port = int(os.environ.get("EVA_CHAT_PORT", os.environ.get("PORT", 8001)))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  EVA · Sala de Conversa em  http://localhost:{port}\n")
    if AUTH_PASSWORD:
        print("  🔒 Login exigido (usuário: eva).\n")
    else:
        print("  ⚠ Sem senha (EVA_PASSWORD não definida) — acesse só por localhost.\n")
    ready = brain_info()
    if ready["ready"]:
        print(f"  Cérebro pronto: {ready['params']:,} parâmetros "
              f"({ready['n_layer']} camadas, contexto {ready['block_size']}).\n")
    else:
        print(f"  {ready['error']}\n")
    print("  Abra no navegador. Ctrl+C para encerrar.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Encerrado.")


if __name__ == "__main__":
    main()
