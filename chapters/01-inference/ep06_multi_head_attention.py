"""推理篇 EP06 · Multi-Head Attention：把 d_model 拆成多个头分别做注意力，再拼回来。

完整实现见 gpt2/model.py 的 MultiHeadAttention（含因果掩码）。
"""
import _common  # noqa: F401
import torch
import torch.nn as nn

from gpt2.model import MultiHeadAttention

vocab = {"猫": 0, "吃": 1, "鱼": 2}
d_model, n_heads = 8, 2

torch.manual_seed(42)
embedding = nn.Embedding(len(vocab), d_model)
x = embedding(torch.tensor([[vocab[w] for w in "猫吃鱼"]]))   # [1, 3, 8]

mha = MultiHeadAttention(d_model, n_heads, max_seq_len=16)
out, weights = mha(x, return_weights=True)

print("输入形状：", x.shape)              # [1, 3, 8]
print("输出形状：", out.shape)            # [1, 3, 8]
print("注意力权重形状：", weights.shape)  # [1, 2, 3, 3]：2 个头，每个头一张 3×3
for h in range(n_heads):
    print(f"\n第 {h} 个头的注意力权重：\n", weights[0, h].detach())
