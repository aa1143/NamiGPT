"""推理篇 EP09 · FFN：升维 → 激活 → 降维。"""
import _common  # noqa: F401
import torch

from gpt2.model import FeedForward

d_model = 4
ffn = FeedForward(d_model, 4 * d_model)   # 中间层是 4 倍：4 -> 16 -> 4

x = torch.tensor([
    [0.9, -0.3, 0.1, -0.7],   # 猫
    [-0.4, 1.0, -0.6, 0.2],   # 吃
    [0.6, -0.5, 0.8, -0.4],   # 鱼
])
hidden = ffn.W1(x)
print("升维后：", hidden.shape)     # [3, 16]
print("降维后：", ffn(x).shape)     # [3, 4]，和输入一致，方便做残差
