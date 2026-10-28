"""推理篇 EP08 · LayerNorm：手写实现，并和 PyTorch 官方对比。"""
import _common  # noqa: F401
import torch
import torch.nn as nn

from gpt2.model import LayerNorm

torch.manual_seed(42)
x = torch.randn(1, 3, 8)   # [batch=1, seq_len=3, d_model=8]

mine = LayerNorm(8)
official = nn.LayerNorm(8)
print("手写和官方的最大差异：", (mine(x) - official(x)).abs().max().item())   # ≈ 0

x_norm = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
print("每行均值（应≈0）：", x_norm.mean(-1))
print("每行方差（应≈1）：", x_norm.var(-1, unbiased=False))
print("每行模长（应≈√8≈2.828）：", x_norm.norm(dim=-1))
