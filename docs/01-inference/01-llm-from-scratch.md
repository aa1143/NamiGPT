# 从零手写大模型 · 推理篇 01 - LLM From Scratch（从零手写大模型）

从零开始，用 PyTorch 手写一个大语言模型，并部署成 ChatGPT 风格的对话接口。

---

## 项目目标

加载 GPT-2 预训练权重，实现完整的推理流程，最终搭建一个可以对话的 FastAPI 后端。

---

## 项目结构

```
llm-from-scratch/
├── docs/          # 学习笔记与操作记录
├── notebooks/     # 实验区，分阶段的 .ipynb 文件
├── src/           # 正式代码
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── model.py
│   └── inference.py
└── api/
    └── main.py    # FastAPI 对话接口
```

---

## 四个核心模块

### 1. Tokenizer（分词器）
把输入文字切分成 Token，再转换成数字 ID，作为模型的输入。

### 2. Transformer 模型
用 PyTorch 手写完整的 Transformer 架构，包括：
- Input Embedding + Positional Encoding
- Multi-Head Attention
- Feed-Forward Network
- Layer Normalization

### 3. 推理引擎
加载 GPT-2 预训练权重，执行 Forward Pass，生成文字输出。

### 4. FastAPI 后端
对外暴露对话接口，配合简单的前端页面，实现 ChatGPT 风格的对话体验。

---

## 技术栈

| 用途 | 工具 |
|---|---|
| 深度学习框架 | PyTorch 2.x |
| 预训练模型 | Hugging Face Transformers（GPT-2） |
| 后端接口 | FastAPI + Uvicorn |
| 开发环境 | Python 3.11 · VS Code · Jupyter Notebook |

---

## 学习进度

- [x] 环境配置
- [ ] Embedding + Positional Encoding
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Feed-Forward Network + Layer Norm
- [ ] 完整 Transformer Forward Pass
- [ ] 加载 GPT-2 权重 · 推理
- [ ] FastAPI 对话接口

---

## 参考资料

- 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 模型：[GPT-2 · Hugging Face](https://huggingface.co/gpt2)
