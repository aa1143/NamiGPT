"""推理篇 EP13 · Loading Pretrained Weights：把 OpenAI 的权重搬进手写模型。

需要：pip install transformers tiktoken
国内网络：export HF_ENDPOINT=https://hf-mirror.com
"""
import _common  # noqa: F401
import tiktoken
import torch
from transformers import GPT2LMHeadModel

from gpt2 import GPT, GPT2_CONFIGS, load_state_dict_from_hf

hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = hf_model.state_dict()

print("官方权重的前几项：")
for name in list(sd)[:6]:
    print(f"  {name:35s} {tuple(sd[name].shape)}")
# 注意 c_attn.weight 是 [768, 2304]：Q、K、V 三个矩阵拼在一起，而且是 [in, out] 方向

model = GPT(GPT2_CONFIGS["gpt2"])
load_state_dict_from_hf(model, sd)
model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = "The cat sat on the"
ids = torch.tensor([enc.encode(prompt)])
with torch.no_grad():
    mine = model(ids)
    official = hf_model(ids).logits

print("\n和官方实现的最大误差：", (mine - official).abs().max().item())
next_id = mine[0, -1].argmax().item()
print(f"{prompt!r} 的下一个词预测：{enc.decode([next_id])!r}")
