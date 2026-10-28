"""推理篇 EP03 · Embedding：Token ID 查表变成向量。"""
import torch
import torch.nn as nn

torch.manual_seed(42)

vocab_size = 30      # 假设词表一共 30 个 Token
embedding_dim = 8    # 每个 Token 用 8 维向量表示（GPT-2 是 768）

# 本质就是一张 30 行 × 8 列的随机数表
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor([3, 15, 27])   # 假设「猫 吃 鱼」的 ID
vectors = embedding_layer(token_ids)

print(vectors.shape)   # torch.Size([3, 8])：3 个 Token，每个 8 维
print(vectors)

# 查表 = 取出权重矩阵的对应行
assert torch.equal(vectors, embedding_layer.weight[token_ids])
print("查表结果和直接取第 3/15/27 行完全一致")
