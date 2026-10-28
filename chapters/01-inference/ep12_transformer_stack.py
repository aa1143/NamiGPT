"""推理篇 EP12 · Transformer Stack：把 12 个 Block 堆起来，就是完整的 GPT-2。"""
import _common  # noqa: F401
import torch

from gpt2 import GPT, GPT2_CONFIGS

config = GPT2_CONFIGS["gpt2"]
model = GPT(config)   # 随机初始化，还不会说人话
print(config)
print(f"参数量：{model.num_params() / 1e6:.1f}M")   # 124.4M

x = torch.tensor([[464, 3797, 25365]])   # "The cat eats"
logits = model(x)
print("logits 形状：", logits.shape)      # [1, 3, 50257]：每个位置都对下一个词打分

print("\n各部分参数量：")
print(f"  token_emb  {model.token_emb.weight.numel():>12,}")
print(f"  pos_emb    {model.pos_emb.weight.numel():>12,}")
print(f"  12 个 Block {sum(p.numel() for p in model.blocks.parameters()):>11,}")
print(f"  ln_f       {sum(p.numel() for p in model.ln_f.parameters()):>12,}")
print("  head       （与 token_emb 共享，不额外占参数）")
