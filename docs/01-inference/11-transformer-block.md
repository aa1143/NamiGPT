# 从零手写大模型 · 推理篇 11 - Transformer Block（Transformer 模块）

## 承接第 10 期：把激活函数放回完整的 Transformer Block

上一期（第 10 期），我们对比了 ReLU 和 GELU，讲清楚了激活函数为什么是 FFN 的关键。第 09 期搭起了 FFN 的骨架，第 10 期补上了其中的非线性机制；这一期，我们把它们和 Attention、LayerNorm、残差连接组装起来。

从第 01 期的项目路线开始，到第 10 期，我们已经陆续介绍并实现了这些组件：

- **Tokenizer / BPE**：文字怎么变成数字
- **Embedding**：数字怎么变成有意义的向量
- **Positional Encoding**：模型怎么知道"谁在前谁在后"
- **Self-Attention**：一个词怎么"看"到其他词
- **Multi-Head Attention**：怎么同时从多个角度看
- **Residual Connection**：怎么保住原始信息不丢
- **LayerNorm**：怎么按 token 归一化并调节特征尺度
- **FFN**：每个位置怎么单独做非线性加工
- **激活函数（ReLU vs GELU）**：非线性从哪来

这些组件分工不同：Tokenizer、Embedding 和位置编码负责输入准备；GPT-2 内部反复堆叠的 **Transformer Block**，则由 Attention、LayerNorm、残差连接和包含激活函数的 FFN 组成——业内管它叫 **Transformer Block**。

今天这一集不引入任何新概念,纯粹是"装配"：把前十期的产出，用一个类装起来。

## 公式:两行,但这两行是全部的秘密

Transformer Block(准确说是GPT-2用的Pre-LN结构)长这样:

```
x1 = x  + Attention(LayerNorm(x))
x2 = x1 + FFN(LayerNorm(x1))
```

这两行信息量很大,拆开看:

1. `LayerNorm(x)` —— 先把x标准化一下,再送进Attention。**注意**:标准化之后的结果只是"喂给Attention看"的,真正做残差相加的时候,用的是原始的、没有标准化过的 `x`,不是 `LayerNorm(x)`。这是Pre-LN和Post-LN最容易搞混的地方。
2. `x + Attention(...)` —— 残差连接,保证Attention就算学得一塌糊涂,原始信息x也不会丢。
3. 第二行同理,只是把Attention换成了FFN。

一句话总结这个Block在干什么:**先做全局信息交换(Attention),再做逐位置的非线性加工(FFN),每一步都留一条"信息高速公路"(残差)绕过去。**

## 猫吃鱼:走一遍完整流程

还是老朋友,"猫吃鱼",3个token,`d_model=4`。假设经过前面几集的处理,我们已经拿到了输入:

```
x.shape = [3, 4]   # 3个token,每个token是4维向量
```

完整走一遍Block内部发生了什么(维度全程标注,数值不重要,形状才是重点):

| 步骤 | 操作 | 输出shape |
|---|---|---|
| 1 | `x_norm1 = LayerNorm(x)` | [3, 4] |
| 2 | `attn_out = MultiHeadAttention(x_norm1)` | [3, 4] |
| 3 | `x1 = x + attn_out` （残差,用的是原始x） | [3, 4] |
| 4 | `x_norm2 = LayerNorm(x1)` | [3, 4] |
| 5 | `ffn_out = FFN(x_norm2)` | [3, 4] |
| 6 | `x2 = x1 + ffn_out` （残差,用的是x1不是x_norm2） | [3, 4] |

看到没有:**从头到尾,shape都是`[3, 4]`,没有变过。** 这不是巧合,而是Transformer Block的设计要求——输入输出形状必须一致,因为GPT-2要把很多个这样的Block直接首尾相连、堆叠起来(比如GPT-2 small堆12层),如果shape会变,根本没法叠。

同样要提醒一句(这句话我们说了很多集了):这里的数值是随机初始化的权重算出来的,没有语义,只是用来验证shape和流程走没走通。真正有意义的数值,要等加载了GPT-2预训练权重之后才谈得上。

## 直观理解:Block是一个"信息加工车间"

可以把一个Transformer Block想象成车间里的一道工序:

- **Attention部分**是"部门间开会"——每个token(员工)都听一听其他token说了什么,更新一下自己的认知。
- **FFN部分**是"回工位单独处理"——开完会,每个人回到自己工位,根据刚才听到的信息,自己做点非线性加工,消化一下。
- **残差连接**是"工牌"——不管开会讨论出什么、工位上加工出什么,每个人始终记得自己原本是谁,不会因为开了个会就"人格分裂"。

GPT-2把这样一道工序重复12次(small版本),每一层都在前一层加工过的基础上,再开一次会、再回工位加工一次。层数越多,能捕捉的关系就越复杂。

## 代码实现

下面沿用第 06 期的 `MultiHeadAttention` 和第 09 期的 `FeedForward` 类；运行前需要先定义或导入这两个类。为衔接第 06 期接口，注意力计算时临时补上 batch 维度，并取出返回值中的输出张量。

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)  # EP06
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)             # EP09

    def forward(self, x):
        # x: [seq_len, d_model]

        # 第一步:Attention + 残差
        x_norm1 = self.ln1(x)              # [seq_len, d_model]
        # EP06 接收 [batch, seq_len, d_model]，返回输出和注意力权重
        attn_out, _ = self.attn(x_norm1.unsqueeze(0))
        attn_out = attn_out.squeeze(0)     # [seq_len, d_model]
        x1 = x + attn_out                  # 残差用原始x,不是x_norm1

        # 第二步:FFN + 残差
        x_norm2 = self.ln2(x1)             # [seq_len, d_model]
        ffn_out = self.ffn(x_norm2)        # [seq_len, d_model]
        x2 = x1 + ffn_out                  # 残差用x1,不是x_norm2

        return x2  # [seq_len, d_model],形状和输入完全一致


# 猫吃鱼验证
d_model, n_heads, d_ff = 4, 2, 16
block = TransformerBlock(d_model, n_heads, d_ff)

x = torch.randn(3, d_model)  # 3个token(猫/吃/鱼),4维embedding
out = block(x)

print(out.shape)  # torch.Size([3, 4]) —— 和输入一模一样
```

注意`MultiHeadAttention`和`FeedForward`这两个类,分别是我们EP06和EP09里已经写过的代码——今天这集没有写任何一行"新"的核心逻辑,全部是**组装**。这也是为什么我说这集是收口:前面实现的 Attention 和 FFN 在这里被组合起来，LayerNorm 与残差连接负责把两条子层路径衔接好；Tokenizer、Embedding 和位置编码仍放在 Block 外部。

## 下一集预告

`TransformerBlock`这个类写完了,GPT-2其实就是把这样的Block堆叠N层(small版本是12层),再加上一个输出层。下一集我们就来做这件事——**把多个Block堆起来,跑通完整的forward pass**,离"加载GPT-2真实预训练权重"就只剩最后一步了。
