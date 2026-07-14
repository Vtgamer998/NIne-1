# 🐉 NINE-1

> **IA de programação em PT-BR, construída do zero.**

![status](https://img.shields.io/badge/status-MVP-orange) ![python](https://img.shields.io/badge/python-3.10+-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## O que é

NINE-1 é uma IA de geração de código **construída do zero** (do tokenizer ao transformer), focada em Python e descrições em **português brasileiro**. Foi pensada para rodar em celular (Termux) e treinar em cloud (Google Colab).

> **Honestidade**: NINE-1 não é Opus nem Fable. É um projeto didático que mostra como um LLM de programação é feito, com um MVP funcional (~10-50M parâmetros) que cabe em qualquer celular moderno.

---

## Estrutura

```
nine-1/
├── nine/
│   ├── __init__.py
│   ├── tokenizer.py    # BPE do zero
│   ├── model.py        # Transformer minimal (RMSNorm, RoPE-friendly, FlashAttn via SDPA)
│   ├── train.py        # loop de pré-treinamento
│   ├── finetune.py     # LoRA (implementação própria, sem peft)
│   ├── quantize.py     # exporta .half / .int8
│   ├── cli.py          # interface em PT-BR
│   ├── data.py         # dataset/memmap
│   ├── prep_data.py    # pré-processa corpus
│   └── data/           # corpus, tokenizer.json, checkpoints
├── scripts/
│   └── download_seed.py
├── notebooks/
│   └── train_nine1.ipynb   # fluxo completo no Google Colab
├── tests/
│   └── test_basic.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Setup local (Termux no celular)

```bash
pkg install python git
git clone <repo> nine-1
cd nine-1
pip install -r requirements.txt
```

### 2) Gera corpus seed e treina tokenizer BPE

```bash
python scripts/download_seed.py
python -m nine.prep_data --paths nine/data/seed --train_bpe --vocab 4096 --max_chars 5000000
```

### 3) Pré-treino (recomendado: Google Colab)

Copie o notebook `notebooks/train_nine1.ipynb` para o seu Drive e execute célula a célula. Tempo: 2-4h na T4 gratuita para ~30M params.

Ou, se tiver GPU local:

```bash
python -m nine.train \
    --data nine/data/corpus.bin \
    --out nine/data/nine1-base.pt \
    --vocab 4096 --block_size 512 \
    --n_layer 10 --n_head 8 --n_embd 512 \
    --batch_size 16 --max_iters 2000 --lr 3e-4
```

### 4) Fine-tuning LoRA (em PT-BR)

```bash
python -m nine.finetune \
    --base nine/data/nine1-base.pt \
    --data nine/data/instruct.jsonl \
    --out nine/data/nine1-instruct.pt \
    --lora_r 8 --lora_alpha 16 --max_iters 200
```

### 5) Use a CLI

```bash
python -m nine.cli "escreva uma funcao fibonacci" --ckpt nine/data/nine1-base.pt --tokens 100
```

Saída:
```
--- NINE-1 ---
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
--------------
```

### 6) Exportar leve (quantização)

```bash
python -m nine.quantize nine/data/nine1-base.pt --mode half --out nine/data/nine1-base.pth
```

---

## Arquitetura (resumo)

| Componente | Implementação |
|------------|----------------|
| Tokenizer | BPE do zero (256 bytes base + merges learned). Regex com suporte a acentos PT. |
| Model | Decoder Transformer pré-norm, RMSNorm, MLP-GELU, multi-head causal self-attention |
| Atenção | `torch.nn.functional.scaled_dot_product_attention` (FlashAttention quando disponível) |
| Otimizador | AdamW (β=0.9, 0.95), cosine LR schedule, weight decay 0.1 |
| LoRA | Implementação própria (PEFT-like), congela base, treina A/B low-rank |
| Quantização | float16 e int8 per-row simétrico |

**Hiperparâmetros padrão (tiny)**: 6 camadas, 6 cabeças, 384 dim, ctx=256 → ~12M params.
**Padrão recomendado (colab)**: 10 camadas, 8 cabeças, 512 dim, ctx=512 → ~30M params.

---

## Limitações (honestas)

* Corpus seed deste MVP é **pequeno** (~5MB). Para resultados reais, alimente com `The Stack`, `CodeAlpaca`, etc.
* Sem pretraining safety/curation: pode gerar código ruim/inseguro.
* Não tem RLHF, DPO ou Constitutional AI (requer meses + GPUs caras).
* Pode "alucinar" funções e APIs. Sempre revise.

---

## Roadmap

- [x] Tokenizer BPE próprio
- [x] Transformer minimal
- [x] Loop de treino + checkpointing
- [x] LoRA com implementação própria
- [x] CLI PT-BR
- [x] Notebook Colab
- [ ] Conversor para GGUF + llama.cpp
- [ ] Avaliação em HumanEval-pt (subset)
- [ ] RLHF leve (DPO) com exemplos PT-BR
- [ ] Interface Gradio web

---

## Créditos

* Inspirado em **nanoGPT** (Andrej Karpathy) — arquitetura e laço de treino.
* **GPT-2 / GPT-3** (Radford et al.) — idéias BPE e pré-norm Transformer.
* **Tutoriais** sobre LoRA (Microsoft Research) para o módulo LoRALinear.

---

## Licença

MIT — use, modifique, distribua. Atribuição apreciada.
