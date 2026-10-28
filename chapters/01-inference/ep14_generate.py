"""推理篇 EP14 · Text Generation：贪心 vs 温度 + top-k 采样。"""
import _common  # noqa: F401
import tiktoken
import torch

from gpt2 import from_pretrained, generate, next_token_candidates

enc = tiktoken.get_encoding("gpt2")
model = from_pretrained("gpt2")

prompt = "The cat sat on the"
ids = torch.tensor([enc.encode(prompt)])

print("下一个词的前 10 个候选：")
for tid, p in next_token_candidates(model, ids, k=10):
    print(f"  {enc.decode([tid])!r:12s} {p:6.2%}")

print("\n【贪心】temperature=0，每次都选概率最大的：")
print(enc.decode(generate(model, ids, max_new_tokens=30, temperature=0)[0].tolist()))

torch.manual_seed(42)
for t in [0.7, 1.2]:
    print(f"\n【采样】temperature={t}, top_k=50：")
    print(enc.decode(generate(model, ids, max_new_tokens=30, temperature=t, top_k=50)[0].tolist()))
