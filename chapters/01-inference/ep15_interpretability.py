"""推理篇 EP15 · Interpretability：打开黑盒看一眼。

1. 看注意力权重：最后一个词在每一层「最关注」前面哪个词
2. Logit Lens：把每一层的中间结果直接接到输出头上，看模型在第几层就「想好了」答案
"""
import _common  # noqa: F401
import tiktoken
import torch

from gpt2 import from_pretrained

enc = tiktoken.get_encoding("gpt2")
model = from_pretrained("gpt2")

prompt = "The cat sat on the"
ids = torch.tensor([enc.encode(prompt)])
tokens = [enc.decode([i]) for i in ids[0].tolist()]

with torch.no_grad():
    x = model.token_emb(ids) + model.pos_emb(torch.arange(ids.size(1)))
    print(f"输入：{tokens}\n")
    print("层   最后一个词最关注的词   该层的 Logit Lens 预测")
    for i, block in enumerate(model.blocks):
        _, w = block.attn(block.ln1(x), return_weights=True)   # [1, heads, T, T]
        focus = tokens[w[0, :, -1].mean(0).argmax().item()]     # 所有头平均后，最后一个位置的注意力
        x = block(x)
        guess = enc.decode([model.head(model.ln_f(x))[0, -1].argmax().item()])
        print(f"{i:>2}   {focus!r:20s}   {guess!r}")
