#!/bin/bash
# NINE-1 Web UI - Script de inicializacao
# Uso: bash run_webui.sh [--share] [--port 7860]
#
# Nota: Requer torch + gradio instalados.
# No Termux (Android), use o Google Colab em vez disso:
#   https://colab.research.google.com/github/Vtgamer998/nine-1/blob/main/notebooks/train_nine1.ipynb

set -e

# Verifica dependencias criticas
MISSING=""
for mod in torch gradio; do
    if ! python -c "import $mod" 2>/dev/null; then
        MISSING="$MISSING $mod"
    fi
done

if [ -n "$MISSING" ]; then
    echo "[erro] Dependencias faltando:$MISSING"
    echo ""
    echo "Para rodar localmente (Linux/desktop):"
    echo "  pip install torch numpy gradio"
    echo ""
    echo "Para rodar no Google Colab (recomendado no celular):"
    echo "  Abra o notebook: notebooks/train_nine1.ipynb"
    echo "  Ou: https://colab.research.google.com/github/Vtgamer998/nine-1/blob/main/notebooks/train_nine1.ipynb"
    exit 1
fi

# Auto-detect: prefere checkpoint grande (-g) se existir
CKPT_BASE="nine/data/nine1-base.pt"
CKPT_G="nine/data/nine1-base-g.pt"
if [ -f "$CKPT_G" ] && [ $(stat -c%s "$CKPT_G") -gt 50000000 ]; then
    CKPT="$CKPT_G"
else
    CKPT="$CKPT_BASE"
fi

TOK_BASE="nine/data/nine1-tok.json"
TOK_G="nine/data/nine1-tok-g.json"
if [ -f "$TOK_G" ]; then
    TOK="$TOK_G"
else
    TOK="$TOK_BASE"
fi

LORA_BASE="nine/data/nine1-instruct.pt"
LORA_G="nine/data/nine1-lora-g.pt"
if [ -f "$LORA_G" ]; then
    LORA="$LORA_G"
else
    LORA="$LORA_BASE"
fi

# Verifica checkpoint
if [ ! -f "$CKPT" ]; then
    echo "[aviso] Checkpoint base nao encontrado: $CKPT"
    echo "[aviso] Use --ckpt para especificar o caminho"
fi

echo "=== NINE-1 Web UI ==="
echo "Checkpoint: $CKPT"
if [ -f "$CKPT" ]; then
    echo "  Size:     $(du -h "$CKPT" | cut -f1)"
fi
echo "Tokenizer:  $TOK"
echo "LoRA:       $LORA"
if [ -f "$LORA" ]; then
    echo "  Size:     $(du -h "$LORA" | cut -f1)"
fi
echo ""

echo "[Dica] Use o modo instruct para melhores resultados!"
echo "  Ex: 'escreva uma funcao fibonacci em python'"
echo ""

python -m nine.webui \
    --ckpt "$CKPT" \
    --tok "$TOK" \
    --lora "$LORA" \
    "$@"
