"""推理篇 EP05 · Self-Attention：单头自注意力的四步计算。"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

d_model = d_k = 8
X = torch.randn(3, d_model)   # 「猫 吃 鱼」经过 Embedding + 位置编码后的输入，[3, 8]

W_q = torch.nn.Linear(d_model, d_k, bias=False)
W_k = torch.nn.Linear(d_model, d_k, bias=False)
W_v = torch.nn.Linear(d_model, d_k, bias=False)
Q, K, V = W_q(X), W_k(X), W_v(X)   # 各 [3, 8]

scores = Q @ K.T                     # 第一步：相似度         [3, 3]
scores = scores / (d_k ** 0.5)       # 第二步：缩放
weights = F.softmax(scores, dim=-1)  # 第三步：归一化成权重，每行和为 1
output = weights @ V                 # 第四步：加权求和       [3, 8]

print("注意力权重（不带掩码，每个词都能看到全句）：\n", weights)

# GPT-2 是「从左往右」生成的，每个词只能看自己和前面的词 → 因果掩码
mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
causal = F.softmax((Q @ K.T / d_k ** 0.5).masked_fill(~mask, float("-inf")), dim=-1)
print("\n加上因果掩码后（上三角全为 0）：\n", causal)
print("输出形状：", (causal @ V).shape)
