"""手写 GPT-2 vs HuggingFace 官方实现：同一份权重、同一个输入，输出必须一致。

默认用随机初始化的官方模型做对比，不需要联网下载权重，几秒钟就能跑完。
设置环境变量 TEST_PRETRAINED=1 会额外下载真实的 GPT-2 124M 权重再对比一次。

运行：
    pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt2 import GPT, GPTConfig, load_state_dict_from_hf  # noqa: E402

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

ATOL = 1e-4


def build_pair(d_model, n_heads, n_layers, max_seq_len=64, vocab_size=50257, seed=0):
    torch.manual_seed(seed)
    hf = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=d_model,
            n_head=n_heads,
            n_layer=n_layers,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    mine = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )
    ).eval()
    load_state_dict_from_hf(mine, hf.state_dict())
    return mine, hf


@pytest.mark.parametrize(
    "d_model,n_heads,n_layers",
    [
        (64, 4, 2),     # 小模型
        (768, 12, 2),   # GPT-2 124M 的宽度，只取 2 层以加快测试
    ],
)
def test_logits_match(d_model, n_heads, n_layers):
    mine, hf = build_pair(d_model, n_heads, n_layers)
    idx = torch.randint(0, 50257, (2, 17))
    with torch.no_grad():
        a = mine(idx)
        b = hf(idx).logits
    assert a.shape == b.shape
    diff = (a - b).abs().max().item()
    assert diff < ATOL, f"最大误差 {diff:.2e}"


def test_causal_mask():
    """改动后面的 token，不应该影响前面位置的输出。"""
    mine, _ = build_pair(64, 4, 2)
    idx = torch.randint(0, 50257, (1, 10))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 50257
    with torch.no_grad():
        a, b = mine(idx), mine(idx2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


@pytest.mark.skipif(os.environ.get("TEST_PRETRAINED") != "1", reason="设置 TEST_PRETRAINED=1 才下载真实权重")
def test_pretrained_gpt2():
    from gpt2 import from_pretrained

    mine = from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    idx = torch.tensor([[464, 3797, 3332, 319, 262]])  # "The cat sat on the"
    with torch.no_grad():
        a = mine(idx)
        b = hf(idx).logits
    diff = (a - b).abs().max().item()
    assert diff < 1e-3, f"最大误差 {diff:.2e}"
    assert a[0, -1].argmax() == b[0, -1].argmax()
