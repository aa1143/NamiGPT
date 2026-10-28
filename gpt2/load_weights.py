"""EP13：把 OpenAI 官方的 GPT-2 权重，搬进我们手写的模型。

官方权重通过 HuggingFace 下载（只用它来拿权重文件，模型计算全部是我们自己写的）。
国内网络可以先设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com

两个容易踩的坑：
1. 转置：GPT-2 原始代码用的是 Conv1D，权重形状是 [in, out]；
   nn.Linear 的权重形状是 [out, in]，所以搬过来要 .T 转置。
2. QKV 合并存储：官方把 W_q、W_k、W_v 拼成了一个 c_attn，形状 [768, 2304]，
   需要沿最后一维切成三份，再分别转置。
"""

import torch

from .config import GPT2_CONFIGS
from .model import GPT


@torch.no_grad()
def load_state_dict_from_hf(model: GPT, sd: dict) -> GPT:
    """把 HuggingFace GPT2LMHeadModel 的 state_dict 拷进手写模型。"""
    d = model.config.d_model

    def copy(dst, src):
        assert dst.shape == src.shape, f"形状对不上：{tuple(dst.shape)} vs {tuple(src.shape)}"
        dst.copy_(src)

    # Embedding：原样拷贝
    copy(model.token_emb.weight, sd["transformer.wte.weight"])
    copy(model.pos_emb.weight, sd["transformer.wpe.weight"])

    for i, block in enumerate(model.blocks):
        p = f"transformer.h.{i}."

        # LayerNorm：原样拷贝
        copy(block.ln1.gamma, sd[p + "ln_1.weight"])
        copy(block.ln1.beta, sd[p + "ln_1.bias"])
        copy(block.ln2.gamma, sd[p + "ln_2.weight"])
        copy(block.ln2.beta, sd[p + "ln_2.bias"])

        # Attention：c_attn 切成 Q/K/V 三份，每份都要转置
        w_q, w_k, w_v = sd[p + "attn.c_attn.weight"].split(d, dim=1)   # 各 [768, 768]
        b_q, b_k, b_v = sd[p + "attn.c_attn.bias"].split(d, dim=0)
        copy(block.attn.W_q.weight, w_q.T)
        copy(block.attn.W_k.weight, w_k.T)
        copy(block.attn.W_v.weight, w_v.T)
        copy(block.attn.W_q.bias, b_q)
        copy(block.attn.W_k.bias, b_k)
        copy(block.attn.W_v.bias, b_v)
        copy(block.attn.W_o.weight, sd[p + "attn.c_proj.weight"].T)
        copy(block.attn.W_o.bias, sd[p + "attn.c_proj.bias"])

        # FFN：同样要转置
        copy(block.ffn.W1.weight, sd[p + "mlp.c_fc.weight"].T)
        copy(block.ffn.W1.bias, sd[p + "mlp.c_fc.bias"])
        copy(block.ffn.W2.weight, sd[p + "mlp.c_proj.weight"].T)
        copy(block.ffn.W2.bias, sd[p + "mlp.c_proj.bias"])

    # 最终 LayerNorm
    copy(model.ln_f.gamma, sd["transformer.ln_f.weight"])
    copy(model.ln_f.beta, sd["transformer.ln_f.bias"])

    # 输出头和 token_emb 共享权重，上面拷 wte 时已经一起拷好了，不用再处理
    return model


def from_pretrained(name: str = "gpt2") -> GPT:
    """下载官方权重并返回手写的 GPT-2。name 可选 gpt2 / gpt2-medium / gpt2-large / gpt2-xl。"""
    from transformers import GPT2LMHeadModel  # 只用来下载权重

    hf_model = GPT2LMHeadModel.from_pretrained(name)
    model = GPT(GPT2_CONFIGS[name])
    load_state_dict_from_hf(model, hf_model.state_dict())
    model.eval()
    return model
