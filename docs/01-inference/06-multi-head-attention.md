# 从零手写大模型 · 推理篇 06 - Multi-Head Attention（多头注意力机制）

[![多头注意力机制计算流程图](https://picui.ogmua.cn/s1/2026/09/08/6a9f9243583ec.webp)](https://picui.ogmua.cn/s1/2026/09/08/6a9f9243583ec.webp)

## 一、上集回顾：单头注意力的局限

上一集我们实现了 Self-Attention 的完整计算流程：Q、K、V 三个矩阵，经过缩放点积，得到了"猫吃鱼"这句话里每个字对其他字的关注程度。

但如果你仔细想一下会发现一个问题：**一次 Attention 计算，只能学到"一种"关注模式**。

"猫吃鱼"这句话里，其实同时存在好几种关系：
- 语法关系：谁是主语，谁是谓语，谁是宾语
- 语义关系：谁对谁做了动作
- 位置关系：谁离谁更近

单头注意力把这些关系全部压缩进同一组 Q/K/V 权重里去学习，相当于让一个人同时兼顾好几件不相关的事，效果注定是打折扣的。

论文原文是这么说的：

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."
> —— *Attention Is All You Need*, Vaswani et al., 2017

大白话翻译：与其用一组权重死磕所有关系，不如**分成几组独立的权重，每组各自学一种关系**，最后把结果拼起来。这就是 Multi-Head Attention。

---

## 二、多头是怎么"多"出来的

这里最容易产生的误解是：多头 = 把 Self-Attention 重复算好几遍，输入输出维度不变。

实际上不是重复计算，而是**切分**。

以我们的例子为例：`d_model = 8`，如果设置 `num_heads = 2`，那么：

| 步骤 | 单头 Self-Attention（EP05） | 多头 Attention（本集） |
|---|---|---|
| Q/K/V 维度 | 每个字一个 8 维向量 | 8 维向量被切成 2 份，每份 4 维 |
| 计算方式 | 一次缩放点积注意力 | 2 组注意力并行计算，互不干扰 |
| 输出 | 直接得到 8 维结果 | 2 个 4 维结果拼接回 8 维 |
| 参数量 | W_q/W_k/W_v 各一个 | 同样是一个 W_q/W_k/W_v，只是切分维度 |

> **关键结论**：多头不是增加了独立的参数矩阵去做多次计算，而是把同一个 d_model 维度拆成若干个子空间（subspace），每个子空间独立跑一次缩放点积注意力，最后拼接、再过一次线性变换。

这里有个硬性约束：`d_model` 必须能被 `num_heads` 整除。我们的例子里 `d_model = 8`，选 `num_heads = 2`，每个头的维度 `d_k = d_model / num_heads = 4`。

---

## 三、论文公式与实现对照

论文给出的多头计算公式是：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

拆解成三步，对照着看会更清楚：

| 论文步骤 | 说明 |
|---|---|
| 每个头独立计算 `Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)` | 每个头有自己的一小块 Q/K/V 权重，各自算一次 EP05 学过的缩放点积注意力 |
| `Concat(head_1, ..., head_h)` | 把所有头的输出在最后一个维度上拼接回原来的 d_model |
| 乘以 `W^O` | 拼接完的结果再过一次线性变换，让不同头的信息重新融合 |

**同样需要划清的边界**：架构设计上"为什么要分头、怎么分头、怎么拼接"是完全可解释的工程决策；但每个头训练完之后具体学到了"语法关系"还是"语义关系"，这是黑盒，模型不会主动告诉你第几个头负责什么，需要事后分析权重才能猜测，而且往往猜不准。

---

## 四、代码实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 切分成多头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力（和 EP05 完全一样的公式，只是多了 num_heads 这一维）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 拼接多头: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(attn_output)
        return output, attn_weights
```

**几个实现细节值得停下来解释一下：**

- `view` 之前必须先 `transpose`，是因为切分维度后，`num_heads` 这一维要提到 `seq_len` 前面，才能让矩阵乘法在"每个头内部"独立进行。
- `contiguous()` 是因为 `transpose` 之后张量在内存里不再连续存储，直接 `view` 会报错，需要先重新整理内存布局。
- `attn_weights` 的 shape 是 `(batch, num_heads, seq_len, seq_len)`，也就是每个头都有一份自己的注意力权重矩阵，这也是后续做可视化分析时最常用到的中间产物。

---

## 五、用"猫吃鱼"跑一遍

沿用一直在用的例子：`vocab = {"猫": 0, "吃": 1, "鱼": 2}`，`d_model = 8`，这次加上 `num_heads = 2`。

```python
vocab = {"猫": 0, "吃": 1, "鱼": 2}
d_model = 8
num_heads = 2

torch.manual_seed(42)
embedding = nn.Embedding(len(vocab), d_model)

sentence = ["猫", "吃", "鱼"]
token_ids = torch.tensor([[vocab[w] for w in sentence]])
x = embedding(token_ids)  # (1, 3, 8)

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output, attn_weights = mha(x)

print("输入 shape:", x.shape)              # torch.Size([1, 3, 8])
print("输出 shape:", output.shape)          # torch.Size([1, 3, 8])
print("注意力权重 shape:", attn_weights.shape)  # torch.Size([1, 2, 3, 3])
```

重点看 `attn_weights` 的 shape：`(1, 2, 3, 3)`。

- `1`：batch size
- `2`：两个头，每个头有自己独立的注意力分布
- `3, 3`：3 个字对 3 个字的关注程度矩阵，这个和 EP05 单头时是一样的

也就是说，"猫"这个字，在头 1 里可能更关注"吃"（谁对我做了动作），在头 2 里可能更关注"鱼"（动作的对象是谁）——这正是我们在第一节里提到的，多头让模型能同时捕捉不同类型的关系。

> 需要说明的是：`d_model = 8`、`num_heads = 2` 这个规模下，两个头到底学到了什么差异化的模式，参数是随机初始化的，没有经过训练，看到的权重分布本身没有语义意义。这里的重点是把**维度切分、并行计算、拼接**这套机制跑通，理解 shape 是怎么变化的；真正"每个头学到不同关系"这件事，要等到后面真正训练模型之后才能观察到。

---

## 六、小结

- 多头注意力不是重复计算多次 Attention，而是把 `d_model` 切成 `num_heads` 份，每份独立做一次缩放点积注意力，再拼接回原维度。
- 硬性约束：`d_model` 必须能被 `num_heads` 整除。
- 架构设计（为什么切分、怎么拼接）是可解释的；训练完之后每个头具体学到了什么关系，是黑盒。
- 本集的例子依然是随机初始化、未经训练的参数，重点是理解结构和 shape 变化，而不是权重的语义。

下一集，我们会把 Multi-Head Attention 的输出接入 Feed-Forward 层，搭出 Transformer Block 的完整前向传播链路。
