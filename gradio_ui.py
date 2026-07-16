"""Interface Gradio da EVA — para hospedar de graça no Hugging Face Spaces.

O Hugging Face só oferece o SDK Docker em hardware pago; o SDK **Gradio**
roda no tier gratuito. Este módulo reconstrói o painel da EVA em Gradio,
reaproveitando toda a lógica de corpus/treino/geração já existente em
`app.py` (nada é reimplementado — só a camada visual muda).

Rodar localmente:  pip install gradio numpy pymupdf && python gradio_ui.py
"""

from __future__ import annotations

import os
import shutil
import time

import gradio as gr

import app  # reutiliza corpus_stats, start_training, generate, etc.

NEON_CSS = """
.gradio-container{background:#05060c!important;color:#eaf2ff!important}
h1,h2,h3{color:#eaf2ff}
.eva-title{font-size:26px;font-weight:800;
  background:linear-gradient(90deg,#22d3ee,#e152ff);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.eva-sub{color:#93a4c4;font-size:14px;margin-top:-6px}
footer{visibility:hidden}
"""


def stats_md() -> str:
    s = app.corpus_stats()
    mats = app.list_materials()
    lines = [f"**{s['chars']:,}** caracteres · **{s['words']:,}** palavras · "
             f"**{s['vocab']}** vocabulário · **{len(mats)}** materiais"]
    if mats:
        lines.append("\n".join(f"- 🧩 {m['name']}  ·  {m['size']:,} B" for m in mats))
    else:
        lines.append("_Nenhum material ainda — alimente a EVA acima._")
    return "\n\n".join(lines)


def add_files(files):
    if files:
        os.makedirs(app.MATERIALS, exist_ok=True)
        for path in files:
            dest = os.path.join(app.MATERIALS, app.safe_name(os.path.basename(path)))
            shutil.copyfile(path, dest)
        app.rebuild_corpus()
    return stats_md()


def add_text(text):
    text = (text or "").strip()
    if len(text) >= 10:
        os.makedirs(app.MATERIALS, exist_ok=True)
        name = f"texto_{int(time.time())}.txt"
        with open(os.path.join(app.MATERIALS, name), "w", encoding="utf-8") as f:
            f.write(text)
        app.rebuild_corpus()
    return stats_md(), ""


def train_stream(preset, tokenizer, steps, dropout, device):
    ok, msg = app.start_training({
        "preset": preset, "tokenizer": tokenizer,
        "steps": int(steps), "dropout": float(dropout),
        "device": device, "bpe_vocab": 512,
    })
    if not ok:
        yield msg
        return
    while app.training_active():
        yield app.read_log() or "> iniciando…"
        time.sleep(1.5)
    yield (app.read_log() or "") + "\n\n✅ Treino concluído."


def stop_train():
    _, msg = app.stop_training()
    return msg


def do_generate(prompt, maxtok):
    return app.generate(prompt or "A ", int(maxtok))


def get_model():
    return app.CKPT if os.path.exists(app.CKPT) else None


def put_model(file):
    if not file:
        return "Selecione um arquivo .pkl."
    if app.training_active():
        return "Pare o treino antes de restaurar."
    if os.path.abspath(file) != os.path.abspath(app.CKPT):
        shutil.copyfile(file, app.CKPT)
    return "✅ Cérebro restaurado — já dá para gerar texto."


def build_demo() -> gr.Blocks:
    app.seed_materials()
    with gr.Blocks(title="EVA · Núcleo Neural") as demo:
        gr.HTML('<div class="eva-title">🧬 EVA · Núcleo Neural</div>'
                '<div class="eva-sub">alimente · treine · converse — sua IA feita do zero</div>')

        with gr.Tab("📚 Memória"):
            stats = gr.Markdown(stats_md())
            refresh = gr.Button("🔄 Atualizar")
            refresh.click(stats_md, outputs=stats)

        with gr.Tab("📡 Alimentar"):
            files = gr.File(label="PDFs / TXT", file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"])
            add_f = gr.Button("+ Adicionar arquivos", variant="primary")
            gr.Markdown("— ou cole um texto —")
            paste = gr.Textbox(label="Texto", lines=4, placeholder="Cole aqui…")
            add_t = gr.Button("+ Injetar texto")
            stats2 = gr.Markdown(stats_md())
            add_f.click(add_files, inputs=files, outputs=stats2)
            add_t.click(add_text, inputs=paste, outputs=[stats2, paste])

        with gr.Tab("⚡ Treinar"):
            with gr.Row():
                preset = gr.Dropdown(["nano", "small", "medium", "large"],
                                     value="small", label="Tamanho do cérebro")
                tokenizer = gr.Dropdown(["char", "bpe"], value="char", label="Percepção")
            with gr.Row():
                steps = gr.Slider(100, 3000, value=1000, step=100, label="Ciclos (passos)")
                dropout = gr.Slider(0.0, 0.6, value=0.1, step=0.05, label="Dropout")
                device = gr.Dropdown(["cpu", "gpu"], value="cpu", label="Processador (gpu = CUDA)")
            with gr.Row():
                btn_train = gr.Button("⚡ Iniciar treino", variant="primary")
                btn_stop = gr.Button("■ Parar")
            log = gr.Textbox(label="Fluxo neural", lines=14, max_lines=14,
                             value="> aguardando ativação…")
            btn_train.click(train_stream, inputs=[preset, tokenizer, steps, dropout, device], outputs=log)
            btn_stop.click(stop_train, outputs=log)

        with gr.Tab("💬 Conversar"):
            prompt = gr.Textbox(label="Semente do pensamento", value="A arte da guerra")
            maxtok = gr.Slider(20, 800, value=250, step=10, label="Extensão (tokens)")
            btn_gen = gr.Button("✨ Gerar", variant="primary")
            out = gr.Textbox(label="EVA responde", lines=6)
            btn_gen.click(do_generate, inputs=[prompt, maxtok], outputs=out)

        with gr.Tab("💾 Salvar / restaurar"):
            gr.Markdown("Baixe o modelo treinado e restaure depois — em qualquer "
                        "aparelho. Seu progresso não se perde se o servidor reiniciar.")
            btn_dl = gr.Button("⬇ Preparar download do cérebro", variant="primary")
            dl = gr.File(label="eva_cerebro.pkl")
            btn_dl.click(get_model, outputs=dl)
            up = gr.File(label="Restaurar cérebro (.pkl)", file_types=[".pkl"])
            up_msg = gr.Markdown()
            up.change(put_model, inputs=up, outputs=up_msg)

    return demo


def launch_demo(**kwargs):
    """Sobe a interface aplicando o tema neon (css/theme via launch no Gradio 6)."""
    theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")
    return build_demo().launch(css=NEON_CSS, theme=theme, **kwargs)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    launch_demo(server_name="0.0.0.0", server_port=port)
