"""推理篇 EP11 · Transformer Block：LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差。"""
import _common  # noqa: F401
import torch

from gpt2 import GPTConfig
from gpt2.model import TransformerBlock

cfg = GPTConfig(vocab_size=30, max_seq_len=16, d_model=4, n_heads=2, n_layers=1)
block = TransformerBlock(cfg)

x = torch.randn(1, 3, cfg.d_model)   # 「猫 吃 鱼」，[1, 3, 4]
out = block(x)
print(out.shape)   # [1, 3, 4]：形状和输入完全一致，所以可以一层层堆叠

print("\n一个 Block 里的参数：")
for name, p in block.named_parameters():
    print(f"  {name:20s} {tuple(p.shape)}")
