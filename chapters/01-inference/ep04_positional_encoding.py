"""推理篇 EP04 · Positional Encoding：给向量加上「位置」信息。

这里演示的是原论文的正余弦位置编码。
注意：GPT-2 实际用的是「可学习的位置嵌入」（一张 1024 × 768 的表，见 gpt2/model.py 的 pos_emb），
思路一样：语义向量 + 位置向量。
"""
import math

import torch

vocab = {"猫": 0, "吃": 1, "鱼": 2}
d_model = 8
max_len = 10


def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()                 # [max_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
    return pe


pe = get_positional_encoding(max_len, d_model)
print("位置 0、1、2 的编码：\n", pe[:3])

torch.manual_seed(42)
embedding = torch.nn.Embedding(len(vocab), d_model)


def encode(tokens):
    ids = torch.tensor([vocab[t] for t in tokens])
    return embedding(ids) + pe[: len(tokens)]   # 语义 + 位置


a = encode(["猫", "吃", "鱼"])
b = encode(["鱼", "吃", "猫"])
print("\n猫吃鱼:\n", a)
print("鱼吃猫:\n", b)
print("\n不加位置编码时，两句话的「猫」向量相同；加了之后：",
      "不同" if not torch.allclose(a[0], b[2]) else "相同")

long_pe = get_positional_encoding(5000, d_model)
print("\n位置编码取值范围：", long_pe.min().item(), "~", long_pe.max().item(), "（恒在 [-1, 1]）")
