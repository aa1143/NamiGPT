"""每个模块的形状检查：输入「猫吃鱼」3 个 token，输出形状要符合文章里的描述。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt2 import GPT, GPT2_CONFIGS, GPTConfig  # noqa: E402
from gpt2.model import (  # noqa: E402
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    gelu,
)

B, T, D = 1, 3, 8


def test_layernorm_matches_pytorch():
    x = torch.randn(B, T, D)
    assert torch.allclose(LayerNorm(D)(x), nn.LayerNorm(D)(x), atol=1e-6)


def test_gelu_is_tanh_approx():
    x = torch.linspace(-5, 5, 101)
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_attention_shapes():
    attn = MultiHeadAttention(D, n_heads=2, max_seq_len=16)
    out, w = attn(torch.randn(B, T, D), return_weights=True)
    assert out.shape == (B, T, D)
    assert w.shape == (B, 2, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, 2, T))   # 每行加起来等于 1
    assert torch.all(w.triu(1) == 0)                         # 看不到未来


def test_ffn_and_block_keep_shape():
    x = torch.randn(B, T, D)
    assert FeedForward(D, 4 * D)(x).shape == x.shape
    cfg = GPTConfig(vocab_size=30, max_seq_len=16, d_model=D, n_heads=2, n_layers=1)
    assert TransformerBlock(cfg)(x).shape == x.shape


def test_gpt_output_shape():
    cfg = GPTConfig(vocab_size=30, max_seq_len=16, d_model=D, n_heads=2, n_layers=2)
    logits = GPT(cfg)(torch.tensor([[3, 15, 27]]))
    assert logits.shape == (1, 3, 30)


def test_gpt2_124m_param_count():
    model = GPT(GPT2_CONFIGS["gpt2"])
    assert model.num_params() == 124_439_808
