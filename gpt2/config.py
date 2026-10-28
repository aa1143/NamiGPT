"""GPT-2 的模型配置。

对应文章：推理篇 EP12（Transformer Stack）。
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257   # 词表大小（EP02 Tokenizer）
    max_seq_len: int = 1024   # 最长上下文，决定位置嵌入表有多少行（EP04）
    d_model: int = 768        # 每个 token 的向量维度（EP03 Embedding）
    n_heads: int = 12         # 注意力头数（EP06）
    n_layers: int = 12        # Transformer Block 层数（EP12）
    layer_norm_eps: float = 1e-5

    @property
    def d_ff(self) -> int:
        # FFN 中间层维度：GPT-2 惯例是 d_model 的 4 倍（EP09）
        return 4 * self.d_model


# OpenAI 发布的四个尺寸
GPT2_CONFIGS = {
    "gpt2":        GPTConfig(d_model=768,  n_heads=12, n_layers=12),   # 124M
    "gpt2-medium": GPTConfig(d_model=1024, n_heads=16, n_layers=24),   # 355M
    "gpt2-large":  GPTConfig(d_model=1280, n_heads=20, n_layers=36),   # 774M
    "gpt2-xl":     GPTConfig(d_model=1600, n_heads=25, n_layers=48),   # 1558M
}
