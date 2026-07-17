"""Testes do modulo DPO (Direct Preference Optimization) para NINE-1."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nine.dpo_train import DPODataset, dpo_collate_fn, compute_dpo_loss
from nine.tokenizer import BPETokenizer


def test_dpo_dataset():
    """Testa criacao e acesso do dataset DPO."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    f.write(json.dumps({
        'prompt': 'escreva fibonacci',
        'chosen': 'def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a',
        'rejected': 'def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)',
    }) + '\n')
    f.write(json.dumps({
        'prompt': 'escreva fatorial',
        'chosen': 'def fat(n):\n    if n <= 1:\n        return 1\n    return n * fat(n-1)',
        'rejected': 'def fat(n):\n    r = 1\n    for i in range(1, n+1):\n        r *= i\n    return r',
    }) + '\n')
    f.close()
    dp = f.name

    bt = BPETokenizer(vocab_size=512)
    bt.train('def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n' * 100, verbose=False)

    ds = DPODataset(dp, block_size=128, tokenizer=bt)
    assert len(ds) == 2, f"Esperado 2 exemplos, obtido {len(ds)}"

    item = ds[0]
    assert len(item) == 4, f"Esperado 4 tensores, obtido {len(item)}"
    for t in item:
        assert isinstance(t, torch.Tensor), f"Esperado Tensor, obtido {type(t)}"

    batch = [ds[i] for i in range(len(ds))]
    collated = dpo_collate_fn(batch)
    assert len(collated) == 4
    assert collated[0].shape[0] == 2  # batch size

    os.unlink(dp)
    print("[ok] test_dpo_dataset")


def test_dpo_loss():
    """Testa computacao da loss DPO com modelo tiny."""
    from nine.model import NINE1, tiny_config

    cfg = tiny_config(vocab_size=512, block_size=128)
    policy = NINE1(cfg)
    ref = NINE1(cfg)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    B, T = 2, 16
    logits_c = torch.randn(B, T, 512, requires_grad=True)
    logits_r = torch.randn(B, T, 512, requires_grad=True)
    labels_c = torch.randint(0, 512, (B, T))
    labels_r = torch.randint(0, 512, (B, T))
    # -100 em posicoes de prompt (primeiras 8)
    labels_c[:, :8] = -100
    labels_r[:, :8] = -100

    ref_c = torch.randn(B, T, 512)
    ref_r = torch.randn(B, T, 512)

    loss, stats = compute_dpo_loss(
        logits_c, labels_c, logits_r, labels_r,
        ref_c, ref_r, beta=0.1,
    )

    assert loss.numel() == 1, f"Loss deve ser escalar, shape={loss.shape}"
    assert loss.requires_grad, "Loss deve ter gradiente"
    assert isinstance(stats, dict), f"Stats deve ser dict, obtido {type(stats)}"
    assert "accuracy" in stats
    assert "margin" in stats
    assert "chosen_reward" in stats
    print(f"  DPO loss: {loss.item():.4f}, acc: {stats['accuracy']:.2f}")
    print("[ok] test_dpo_loss")


def test_dpo_loss_nan_guard():
    """Testa que NaN guard nao quebra com inputs extremos."""
    from nine.dpo_train import compute_dpo_loss

    B, T = 2, 8
    logits_c = torch.full((B, T, 512), 1e10, dtype=torch.float32)
    logits_r = torch.full((B, T, 512), -1e10, dtype=torch.float32)
    labels_c = torch.randint(0, 512, (B, T))
    labels_r = torch.randint(0, 512, (B, T))
    labels_c[:, :4] = -100

    loss, stats = compute_dpo_loss(
        logits_c, labels_c, logits_r, labels_r,
        logits_c, logits_r, beta=100.0,
    )
    assert torch.isfinite(loss).all(), f"Loss deve ser finita, obtido {loss}"
    print(f"  NaN guard test: loss={loss.item():.4f}")
    print("[ok] test_dpo_loss_nan_guard")


if __name__ == "__main__":
    test_dpo_dataset()
    test_dpo_loss()
    test_dpo_loss_nan_guard()
    print("\n=== Todos os testes DPO passaram! ===")
