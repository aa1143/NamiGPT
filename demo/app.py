"""在线体验：用手写的 GPT-2 续写文本，并显示每一步的候选词概率。

运行：
    pip install gradio
    python demo/app.py
然后浏览器打开 http://127.0.0.1:7860

注意：GPT-2 没有经过对话微调，它只会「接着往下写」，不会回答问题；
它的训练语料以英文为主，建议输入英文。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpt2 import from_pretrained  # noqa: E402
from gpt2.generate import EOS_TOKEN_ID, top_k_filter  # noqa: E402

enc = tiktoken.get_encoding("gpt2")
model = from_pretrained("gpt2")


def show(token_id: int) -> str:
    # 用 ␣ 显示空格，方便看出 GPT-2 的 token 往往带着前导空格
    return repr(enc.decode([token_id]))[1:-1].replace(" ", "␣") or "∅"


@torch.no_grad()
def run(prompt, max_new_tokens, temperature, top_k, seed):
    if not prompt.strip():
        return "", "请输入一段开头。"
    torch.manual_seed(int(seed))
    ids = torch.tensor([enc.encode(prompt)])
    steps = []

    for _ in range(int(max_new_tokens)):
        logits = model(ids[:, -model.config.max_seq_len:])[0, -1]

        # 展示用：温度调整后的原始概率（未做 top-k 截断）
        shown = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        top = torch.topk(shown, 5)

        if temperature == 0:
            next_id = logits.argmax().view(1)
        else:
            probs = F.softmax(top_k_filter(logits / temperature, int(top_k)), dim=-1)
            next_id = torch.multinomial(probs, 1)

        steps.append((next_id.item(), list(zip(top.indices.tolist(), top.values.tolist()))))
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == EOS_TOKEN_ID:
            break

    text = enc.decode(ids[0].tolist())

    lines = ["| 步 | 选中 | 候选 Top-5（概率） |", "|---|---|---|"]
    for i, (chosen, cands) in enumerate(steps, 1):
        cells = " · ".join(
            (f"**{show(t)}** {p:.1%}" if t == chosen else f"{show(t)} {p:.1%}") for t, p in cands
        )
        lines.append(f"| {i} | `{show(chosen)}` | {cells} |")
    return text, "\n".join(lines)


with gr.Blocks(title="NamiGPT · 零基础入门大模型") as app:
    gr.Markdown(
        "# NamiGPT · 推理演示\n"
        "模型代码 100% 手写（见 `gpt2/model.py`），权重来自 OpenAI 官方 GPT-2 124M。\n"
        "GPT-2 只会续写、不会对话，建议输入英文开头。"
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="开头", value="The cat sat on the", lines=3)
            max_new = gr.Slider(1, 100, value=20, step=1, label="生成多少个 token")
            temp = gr.Slider(0, 2, value=0.8, step=0.05, label="温度（0 = 贪心）")
            topk = gr.Slider(1, 200, value=50, step=1, label="Top-k")
            seed = gr.Number(value=42, label="随机种子", precision=0)
            btn = gr.Button("生成", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="续写结果", lines=6)
    table = gr.Markdown()
    btn.click(run, [prompt, max_new, temp, topk, seed], [out, table])

if __name__ == "__main__":
    app.launch()
