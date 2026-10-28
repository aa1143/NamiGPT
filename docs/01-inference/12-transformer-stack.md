# 从零手写大模型 · 推理篇 12 - Transformer Stack（Transformer 堆叠）

## 上集回顾

EP11里我们把Pre-LN的两行公式装进了一个`TransformerBlock`：

```
x1 = x + Attention(LayerNorm(x))
x2 = x1 + FFN(LayerNorm(x1))
```

用"猫吃鱼"（`d_model=4`）走了一遍完整的shape标注，最后拼出了一个能跑的`nn.Module`。结尾留了个悬念：**真正的GPT不是一个block，是一摞block**。这一集就来解决这个问题——把单个工位，变成一整条流水线。

## 为什么要堆叠

一个`TransformerBlock`做的事情，说白了就是"每个token看看别的token，再自己消化一下"，做一轮这样的操作。

问题是：一轮消化能学到的东西有限。"猫吃鱼"这种三个字的句子看着简单，但语言里真实的依赖关系可能隔得很远——代词指代谁、前面埋的伏笔在后面才兑现、一个词的含义要看整段上下文才能确定。一层Attention能捕捉到的，往往是比较直接的关联；更抽象、更远距离的关系，需要**多层叠加**，一层一层把理解往深处推。

这也是"深度学习"里"深度"这两个字的来源——不是靠一层算得多准，是靠层数堆出来的。GPT-2 small用了12层，GPT-3更是堆到了96层。层数越多，模型能表达的抽象层次理论上越丰富，当然参数量和计算量也跟着涨。

## 堆叠这件事，公式上其实很朴素

先看一个事实：`TransformerBlock`的输入输出shape是**完全一样的**。

输入`x`是`(batch, seq_len, d_model)`，经过Attention、经过FFN、经过两次残差相加，输出还是`(batch, seq_len, d_model)`。形状没变。

这个"形状不变"是能堆叠的前提——正因为一个block吐出来的东西，长得跟它吃进去的东西一模一样，所以可以直接把上一个block的输出，原封不动喂给下一个block当输入，一直串下去：

```
x0 = 输入(embedding之后)
x1 = Block1(x0)
x2 = Block2(x1)
x3 = Block3(x2)
...
xN = BlockN(x_{N-1})
```

每一层都在上一层的输出基础上继续"消化"，但接口始终不变。第 07 期讲过，残差连接为信息和梯度提供了直接通路，有助于深层网络训练。堆叠时还必须让每个 Block 保持相同的输入输出维度；在这个接口约束下，多个 Block 才能直接串联。层数仍受显存、计算量和训练稳定性的限制。

## 猫吃鱼，走一遍两层堆叠

继续用老朋友"猫吃鱼"，`d_model=4`，3个token，这次堆两层block（`n_layers=2`）方便手算，实际GPT-2 small是12层，公式完全一样，只是循环次数不同。

**输入**（EP03/EP04的Embedding + 位置编码之后）：

```
x0.shape = (1, 3, 4)   # batch=1, seq_len=3(猫/吃/鱼), d_model=4
```

**进第1层**：

```
x1 = Block1(x0)
x1.shape = (1, 3, 4)   # 形状不变，内容已经被Attention+FFN加工过一轮
```

**进第2层**：

```
x2 = Block2(x1)
x2.shape = (1, 3, 4)   # 还是不变
```

到这里，"猫吃鱼"这三个token已经经过了两轮"互相看看再消化"。第一层里，"鱼"这个token可能已经开始关注"吃"；到了第二层，"吃"这个已经带上"鱼"信息的token，又能反过来影响"猫"——层数越多，token之间能建立的间接关联就越复杂。

**最后一步，也是这集要补的一块拼图**：前十一期我们完成了输入准备，并组装了单个 Transformer Block，但一个完整的GPT，在最后一层block之后，还有两个小尾巴：

```
x_final = LayerNorm(x2)          # 再做一次LayerNorm，稳定输出分布
logits  = Linear(x_final)        # (1, 3, 4) -> (1, 3, vocab_size)
```

这个`Linear`就是"输出头"（output head），作用是把`d_model=4`维的向量，重新映射回词表大小的分数（GPT-2词表是50257）。这一步做完，`logits`里每个位置的4个数，就变成了50257个数——对应词表里每一个token的"打分"，分数越高代表模型觉得这个位置接下来越可能是这个token。

## 一个容易漏掉的点：堆叠的是"层"，不是"参数"

新手很容易以为12层是同一个block重复用12次，参数共享。**不是的**——每一层的Attention、FFN都有自己独立的一套权重，12层就是12套完全不同的参数。代码里体现出来就是：

```python
self.blocks = nn.ModuleList([
    TransformerBlock(d_model, n_heads, 4 * d_model)
    for _ in range(n_layers)
])
```

这里用列表推导式创建了`n_layers`个**独立**的`TransformerBlock`实例，每个实例内部的`nn.Linear`初始化时都会拿到自己的随机权重。用`nn.ModuleList`而不是普通Python列表来装，是因为`nn.ModuleList`会让PyTorch自动把里面每个block的参数都注册进整个模型的参数列表——这样调用`model.parameters()`才能拿到全部12层的参数，普通list做不到这件事。

## 流水线的比喻

EP11把单个block比作"组装车间"的一个工位。这次很自然地往下延伸：**多个工位串起来，就是一整条流水线。**

原材料（embedding之后的向量）从流水线入口进去，经过工位1加工，半成品传给工位2，工位2再加工，传给工位3……每个工位做的操作是同一类型的活（Attention+FFN两道工序），但每个工位上的工人（权重）是不同的人，各自负责在前一个工位的基础上再往前推进一点。走到流水线尽头，最后再经过质检+贴标签（LayerNorm+输出头），产出的就不再是半成品向量，而是"词表里每个候选词的打分"——一个可以直接拿去做预测的东西。

## 完整实现

第 11 期为了演示单句话，Block 接收 `[seq_len, d_model]`。这一期统一改为 `[batch, seq_len, d_model]`，因此直接把三维张量传给第 06 期的注意力层，并解包输出和注意力权重，不再临时添加 batch 维。运行前需先定义或导入第 06 期的 `MultiHeadAttention` 和第 09 期的 `FeedForward`。

另外，第 04 期使用固定正余弦位置编码；这里为后续对齐 GPT-2，改用可学习的位置嵌入。当前仍是教学用的堆叠与 shape 演示，第 06 期注意力尚未加入因果掩码，不能直接作为完整的 GPT-2 自回归实现。

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        d_ff = 4 * d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)   # EP03
        self.pos_emb = nn.Embedding(max_seq_len, d_model)    # EP04：这里改用可学习的位置嵌入
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)  # EP11
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)                    # EP08
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)   # (B, T, d_model)

        for block in self.blocks:
            x = block(x)                                # 一层一层往下传，形状不变

        x = self.ln_f(x)
        logits = self.head(x)                           # (B, T, vocab_size)
        return logits
```

核心就是那个`for block in self.blocks`循环——朴素到没什么好讲的，这也正说明了残差连接在架构设计上的价值：**正是因为每一层接口一致，堆叠这件事才能简单到只是一个for循环**。

跑一下shape，验证没问题：

```python
model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
x = torch.tensor([[464, 3797, 25365]])   # "The cat eats"的token id，长度3
logits = model(x)
print(logits.shape)   # torch.Size([1, 3, 50257])
```

⚠️ 老规矩要提醒一句：这里的权重还是随机初始化的，`logits`里的打分现在完全没有语义意义，softmax出来选哪个token基本等于随机数生成器。想让这套结构真正"理解"语言、给出靠谱预测，还差最后一步——把GPT-2训练好的真实权重灌进去。

## 下集预告

EP13我们去干一件事：把HuggingFace上GPT-2训练好的参数，一个个对应搬进我们自己写的这个`GPT`类里。这里边有个坑——HuggingFace存权重的方式（`Conv1D`）和我们用的`nn.Linear`，矩阵是**转置**关系，直接复制会形状不匹配。把这个坑填平，模型才算真正"活"过来。
