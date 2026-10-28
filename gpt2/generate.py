"""EP14：自回归文本生成。

每一步：把当前整句送进模型 → 取最后一个位置的 logits → 选出下一个 token → 拼到句尾 → 重复。
"""

import torch
import torch.nn.functional as F

EOS_TOKEN_ID = 50256  # GPT-2 的 <|endoftext|>


def top_k_filter(logits, k: int):
    """只保留分数最高的 k 个候选，其余设成 -inf（softmax 后概率为 0）。"""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: int = EOS_TOKEN_ID,
):
    """
    input_ids:   [B, T] 的 token id
    temperature: 0 表示贪心（每次都选概率最大的）；越大越随机
    top_k:       只在前 k 个候选里抽样
    """
    model.eval()
    max_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        context = input_ids[:, -max_len:]          # 超过 1024 就只保留最近的部分
        logits = model(context)[:, -1, :]          # [B, vocab_size]

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = top_k_filter(logits / temperature, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # [B, 1]

        input_ids = torch.cat([input_ids, next_id], dim=1)
        if eos_token_id is not None and (next_id == eos_token_id).all():
            break

    return input_ids


@torch.no_grad()
def next_token_candidates(model, input_ids, k: int = 10, temperature: float = 1.0):
    """返回下一个 token 的前 k 个候选及其概率，用于观察模型「在犹豫什么」。"""
    model.eval()
    logits = model(input_ids[:, -model.config.max_seq_len:])[0, -1]
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    top = torch.topk(probs, k)
    return list(zip(top.indices.tolist(), top.values.tolist()))
