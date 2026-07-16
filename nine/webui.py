"""
NINE-1 Web UI — Interface Gradio para chat com IA de programacao PT-BR.

Uso:
    python -m nine.webui --ckpt nine/data/nine1-base.pt --tok nine/data/nine1-tok.json
    python -m nine.webui --ckpt nine/data/nine1-base.pt --lora nine/data/nine1-instruct.pt --share

Dependencias:
    pip install gradio torch numpy
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional, Generator, Tuple

import torch

from .tokenizer import BPETokenizer, BOS_TOKEN, EOS_TOKEN
from .fuse import load_fused_model

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 2048
MAX_NEW_TOKENS = 512
TITLE = "🐉 NINE-1 — IA de Programação em PT-BR"
DESCRIPTION = """
**NINE-1** é uma IA de geração de código construída do zero, 
especializada em Python e descrições em **português brasileiro**.

Digite um prompt em PT-BR para gerar código Python!
"""
THEME = "soft"  # Gradio theme

CSS = """
<style>
    .gradio-container { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
    .chat-message { border-radius: 12px !important; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
                    color: white !important; }
    .bot-message { background: #1a1a2e !important; border: 1px solid #667eea !important; 
                   color: #e0e0e0 !important; }
    code { background: #2d2d2d !important; padding: 2px 6px !important; 
           border-radius: 4px !important; font-size: 0.9em; }
    pre { background: #1a1a2e !important; border: 1px solid #333 !important; 
          border-radius: 8px !important; padding: 12px !important; overflow-x: auto !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Engine de geracao
# ---------------------------------------------------------------------------

class NINE1Engine:
    """Wrapper do modelo NINE-1 para uso no Gradio."""

    def __init__(
        self,
        ckpt_path: str,
        lora_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
        lora_r: int = 8,
        lora_alpha: int = 16,
        verbose: bool = False,
    ):
        self.device = device
        self.model, self.tokenizer = load_fused_model(
            base_path=ckpt_path,
            lora_path=lora_path,
            tokenizer_path=tokenizer_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            verbose=verbose,
        )
        self.model.eval()
        self.use_bpe = self.tokenizer is not None

        if verbose:
            n_params = self.model.num_params()
            print(f"[webui] Modelo pronto: {n_params/1e6:.2f}M params")

    def encode(self, text: str) -> torch.Tensor:
        """Codifica texto para tensor de tokens."""
        if self.use_bpe:
            ids = self.tokenizer.encode(text, add_bos=True)
            ids = ids[-MAX_CONTEXT_TOKENS:]
            return torch.tensor([ids], dtype=torch.long, device=self.device)
        # Fallback
        ids = [min(ord(c), 65535) for c in text[-MAX_CONTEXT_TOKENS:]]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, token_id: int) -> str:
        """Decodifica um token ID para texto."""
        if self.use_bpe:
            return self.tokenizer.decode([token_id])
        if 0 < token_id <= 0x10FFFF:
            try:
                ch = chr(token_id)
                if ch.isprintable() or ch in ("\n", "\t", " "):
                    return ch
            except (ValueError, OverflowError):
                pass
        return ""

    def is_eos(self, token_id: int) -> bool:
        """Verifica se token e EOS."""
        if self.use_bpe and EOS_TOKEN in self.tokenizer.token_to_id:
            return token_id == self.tokenizer.token_to_id[EOS_TOKEN]
        return token_id in (0, 1, 2, 3)

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = None,
        max_tokens: int = 256,
        format_instruct: bool = False,
    ) -> Generator[str, None, None]:
        """Gera tokens um a um com streaming.

        Args:
            prompt: Texto de entrada.
            temperature: Temperatura de amostragem.
            top_k: Top-K sampling (0 para desligar).
            top_p: Top-P nucleus sampling (None para desligar).
            max_tokens: Maximo de tokens a gerar.
            format_instruct: Se True, envolve prompt no formato # tarefa / # solucao.

        Yields:
            Strings parciais do texto gerado.
        """
        prompt = prompt.strip()
        if not prompt:
            yield "Por favor, digite um prompt valido."
            return

        # Formata para instruct mode se solicitado
        if format_instruct:
            formatted_prompt = f"# tarefa: {prompt}\n# solucao:\n"
        else:
            formatted_prompt = prompt

        # Codifica prompt
        input_ids = self.encode(formatted_prompt)
        if input_ids.numel() == 0:
            yield "Erro ao codificar prompt."
            return

        ids = input_ids.clone()
        tokens_generated = 0
        max_tokens = min(max_tokens, MAX_NEW_TOKENS)

        for _ in range(max_tokens):
            with torch.no_grad():
                if ids.size(1) > MAX_CONTEXT_TOKENS:
                    ids = ids[:, -MAX_CONTEXT_TOKENS:]

                logits, _, _ = self.model(ids)
                logits = logits[:, -1, :] / max(temperature, 0.01)

                # Top-K
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-P (nucleus)
                if top_p and top_p > 0.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = sorted_logits.softmax(-1).cumsum(-1)
                    sorted_logits[cum > top_p] = float("-inf")
                    logits.scatter_(1, sorted_idx, sorted_logits)

                probs = logits.softmax(-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            # Decodifica
            token_str = self.decode(next_id.item())
            if token_str:
                yield token_str

            tokens_generated += 1

            # Stop conditions
            if self.is_eos(next_id.item()):
                break
            if tokens_generated >= max_tokens:
                break

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = None,
        max_tokens: int = 256,
        format_instruct: bool = False,
    ) -> str:
        """Gera texto completo (sem streaming)."""
        return "".join(self.generate_stream(
            prompt, temperature, top_k, top_p, max_tokens,
            format_instruct=format_instruct,
        ))


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------

def create_ui(engine: NINE1Engine, args) -> None:
    """Cria e lanca a interface Gradio."""
    import gradio as gr

    def chat_fn(message: str, history: list, temperature: float,
                top_k: int, top_p: float, max_tokens: int,
                mode: str = "chat"):
        """Funcao de chat chamada pelo Gradio.

        Args:
            message: Mensagem do usuario.
            history: Historico do chat.
            temperature: Temperatura de geracao.
            top_k: Top-K sampling.
            top_p: Top-P (nucleus) sampling.
            max_tokens: Maximo de tokens a gerar.
            mode: Modo de prompt ("chat" ou "instruct").

        Yields:
            history_atualizado (sem partial) para streaming.
        """
        if not message or not message.strip():
            yield history
            return

        if mode == "instruct":
            prompt = message
            format_instruct = True
        else:
            system_prompt = (
                "Voce e uma IA de programacao em portugues chamada NINE-1. "
                "Responda com codigo Python quando apropriado. "
                "Sempre gere codigo seguro e bem escrito."
            )

            context = system_prompt + "\n\n"
            if history:
                recent = history[-6:]
                for user_msg, bot_msg in recent:
                    if user_msg:
                        context += f"Usuario: {user_msg}\n"
                    if bot_msg:
                        context += f"NINE-1: {bot_msg}\n"

            context += f"Usuario: {message}\nNINE-1:"
            prompt = context
            format_instruct = False

        adjusted_max = min(max_tokens, MAX_NEW_TOKENS)
        top_p_val = top_p if top_p > 0 else None

        history.append([message, ""])
        partial = ""

        try:
            for token_chunk in engine.generate_stream(
                prompt=prompt,
                temperature=temperature,
                top_k=int(top_k) if top_k > 0 else None,
                top_p=top_p_val,
                max_tokens=adjusted_max,
                format_instruct=format_instruct,
            ):
                partial += token_chunk
                history[-1][1] = partial
                yield history
        except Exception as e:
            error_msg = f"\n\n[Erro na geracao: {e}]"
            partial += error_msg
            history[-1][1] = partial
            yield history

    # --- Layout ---
    with gr.Blocks(
        title=TITLE,
        theme=THEME,
        css=CSS,
    ) as demo:
        gr.HTML(f"""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 1rem;">
            <h1 style="color: white; margin: 0; font-size: 2.2rem;">🐉 NINE-1</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0; font-size: 1.1rem;">
                IA de Programação em Português
            </p>
            <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0; font-size: 0.9rem;">
                {engine.model.num_params()/1e6:.1f}M parâmetros | PoC construída do zero
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Chat area
                chatbot = gr.Chatbot(
                    label="💬 Chat NINE-1",
                    placeholder="Digite um prompt em português para gerar código...",
                    height=500,
                    bubble_full_width=False,
                    render=False,
                )

                # Input area
                with gr.Group():
                    msg = gr.Textbox(
                        label="Seu prompt",
                        placeholder="Ex: escreva uma função fibonacci em python",
                        lines=2,
                        max_lines=6,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Gerar", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Limpar", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parâmetros")
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    label="Temperatura",
                    info="Mais alto = mais criativo",
                )
                top_k = gr.Slider(
                    minimum=0, maximum=100, value=40, step=1,
                    label="Top-K",
                    info="Amostra dos K tokens mais prováveis (0 = desligado)",
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P (nucleus)",
                    info="Probabilidade acumulada",
                )
                max_tokens = gr.Slider(
                    minimum=16, maximum=MAX_NEW_TOKENS, value=200, step=16,
                    label="Máx. tokens",
                    info="Tamanho máximo da resposta",
                )

                gr.Markdown("---")
                gr.Markdown("### 📋 Modo")
                mode = gr.Radio(
                    choices=["instruct", "chat"],
                    value="chat",
                    label="Modo de prompt",
                    info="instruct: formato tarefa/solução",
                )

        gr.Examples(
            examples=[
                ["escreva uma função fibonacci em python"],
                ["crie uma classe Pilha em python"],
                ["escreva uma função que valida email"],
                ["faça um bubble sort em python"],
                ["como ler um arquivo CSV em python?"],
                ["crie uma função que calcula o IMC"],
            ],
            inputs=msg,
            label="📌 Exemplos rápidos",
        )

        with gr.Accordion("📖 Sobre o NINE-1", open=False):
            gr.Markdown("""
            **NINE-1** é uma IA de geração de código **construída do zero** 
            em PyTorch — do tokenizer BPE ao Transformer decoder — focada 
            em **português brasileiro**.

            **Arquitetura:**
            - Tokenizer BPE byte-level (GPT-2 style)
            - Transformer decoder com RoPE + RMSNorm + FlashAttention
            - Fine-tuning LoRA para instruções em PT-BR
            - KV Cache para geração eficiente

            **Limitações (honestas):**
            - Modelo pequeno (~10-50M params) — não é GPT-4
            - Corpus de treino limitado
            - Pode gerar código incorreto — sempre revise
            """)

        gr.HTML("""
        <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.85rem;">
            Feito do zero com PyTorch 🧠 | 
            <a href="https://github.com/Vtgamer998/nine-1" target="_blank">GitHub</a>
        </div>
        """)

        # --- Event handlers ---
        inputs = [msg, chatbot, temperature, top_k, top_p, max_tokens, mode]
        outputs = [chatbot, msg]

        def submit_and_clear(message, chat_history, temp, k, p, tokens, md):
            """Gera, atualiza chatbot, e limpa o input."""
            gen = chat_fn(message, chat_history, temp, k, p, tokens, md)
            for history_state in gen:
                yield history_state, ""

        # Submit (textbox enter)
        msg.submit(
            submit_and_clear, inputs, outputs,
            queue=True,
        )

        # Botao submit
        submit_btn.click(
            submit_and_clear, inputs, outputs,
            queue=True,
        )

        # Clear
        clear_btn.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    # --- Launch ---
    print(f"\nIniciando NINE-1 Web UI...")
    print(f"  URL local: http://127.0.0.1:{args.port}")
    if args.share:
        print(f"  URL publica: (gerada pelo Gradio)")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NINE-1 Web UI (Gradio)")
    p.add_argument("--ckpt", type=str, default="nine/data/nine1-base.pt",
                   help="Checkpoint base .pt")
    p.add_argument("--lora", type=str, default=None,
                   help="Checkpoint LoRA opcional .pt")
    p.add_argument("--tok", type=str, default="nine/data/nine1-tok.json",
                   help="Tokenizer BPE .json")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    p.add_argument("--port", type=int, default=7860, help="Porta")
    p.add_argument("--share", action="store_true", help="Criar link publico")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Validacoes
    if not os.path.exists(args.ckpt):
        print(f"[webui] Erro: checkpoint nao encontrado: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"Inicializando NINE-1 Web UI...")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  LoRA: {args.lora or 'nenhum'}")
    print(f"  Device: {args.device}")

    # Carrega modelo
    engine = NINE1Engine(
        ckpt_path=args.ckpt,
        lora_path=args.lora,
        tokenizer_path=args.tok,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        verbose=args.verbose,
    )

    # Cria UI
    create_ui(engine, args)


if __name__ == "__main__":
    main()
