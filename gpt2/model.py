"""从零手写 GPT-2。

整个模型只用到 PyTorch 最基础的积木：nn.Linear、nn.Embedding 和张量运算。
每个模块都标注了对应的文章编号，可以对照 docs/01-inference/ 阅读。

数据流（以 "猫吃鱼" 为例，B=1, T=3）：

    token ids          [B, T]
      ↓ token_emb + pos_emb                         EP03 / EP04
    x                  [B, T, d_model]
      ↓ × n_layers 个 TransformerBlock               EP11 / EP12
      │    x = x + Attention(LayerNorm(x))           EP05 / EP06 / EP07 / EP08
      │    x = x + FFN(LayerNorm(x))                 EP09 / EP10
      ↓ ln_f
      ↓ head (与 token_emb 共享权重)
    logits             [B, T, vocab_size]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """EP08：层归一化。对每个 token 的向量单独做「减均值、除标准差」，再乘 gamma 加 beta。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 注意：用有偏方差，和 PyTorch 官方一致
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def gelu(x):
    """EP10：GELU 激活函数。

    GPT-2 用的是 tanh 近似版：
        0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    等价于 F.gelu(x, approximate="tanh")。
    注意：PyTorch 的 nn.GELU() 默认是「精确版」（基于 erf），和 GPT-2 有细微差别。
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class MultiHeadAttention(nn.Module):
    """EP05 / EP06：带因果掩码的多头自注意力。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 因果掩码：下三角矩阵。第 i 个 token 只能看到第 0..i 个 token，看不到「未来」。
        # 例如 T=3（猫 吃 鱼）：
        #   猫 [1 0 0]
        #   吃 [1 1 0]
        #   鱼 [1 1 1]
        # 不加这一步，GPT-2 在预训练时就是「偷看答案」的，加载官方权重后输出会错。
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, return_weights: bool = False):
        B, T, d_model = x.shape

        # 1. 线性投影得到 Q、K、V                               [B, T, d_model]
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        # 2. 拆成多头：[B, T, d_model] -> [B, n_heads, T, d_k]
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 缩放点积：Q·Kᵀ / √d_k                              [B, n_heads, T, T]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # 4. 因果掩码：把「未来」位置填成 -inf，softmax 之后就是 0
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        # 5. softmax 得到注意力权重，每一行加起来等于 1
        weights = F.softmax(scores, dim=-1)

        # 6. 加权求和                                           [B, n_heads, T, d_k]
        out = weights @ V

        # 7. 拼接多头 + 输出投影                                [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        out = self.W_o(out)

        return (out, weights) if return_weights else out


class FeedForward(nn.Module):
    """EP09：前馈网络。先升维到 4 倍，过激活函数，再降回来。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)   # 升维：768 -> 3072
        self.W2 = nn.Linear(d_ff, d_model)   # 降维：3072 -> 768

    def forward(self, x):
        return self.W2(gelu(self.W1(x)))


class TransformerBlock(nn.Module):
    """EP11：一个 Transformer Block。GPT-2 用的是 Pre-LN：先 LayerNorm，再进子层，最后残差相加（EP07）。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.max_seq_len)
        self.ln2 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.ffn = FeedForward(config.d_model, config.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差加的是原始 x，不是 ln1(x)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """EP12：把 Embedding、N 个 Block、最终 LayerNorm 和输出头串起来。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)   # EP03
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)    # EP04：GPT-2 用可学习的位置嵌入
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.d_model, config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享（weight tying）：输出头和 token embedding 是同一张表（EP13）
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # GPT-2 的初始化：正态分布，标准差 0.02
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过上限 {self.config.max_seq_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)   # [B, T, d_model]

        for block in self.blocks:
            x = block(x)                               # 每一层形状都不变

        x = self.ln_f(x)
        return self.head(x)                            # [B, T, vocab_size]

    def num_params(self) -> int:
        # 共享的权重只算一次
        return sum(p.numel() for p in self.parameters())
