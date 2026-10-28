"""推理篇 EP02 · Tokenizer：文字 → 字节 → Token ID。

需要：pip install tiktoken（首次运行会下载 GPT-2 的词表）
"""
import tiktoken

enc = tiktoken.get_encoding("gpt2")
print("GPT-2 词表大小：", enc.n_vocab)  # 50257

for text in ["猫吃鱼", "cat eats fish", "The cat eats"]:
    ids = enc.encode(text)
    print(f"\n{text!r}")
    print("  UTF-8 字节：", text.encode("utf-8").hex(" ").upper(), f"（{len(text.encode('utf-8'))} 个字节）")
    print("  Token ID ：", ids, f"（{len(ids)} 个 Token）")
    print("  每个 Token 对应的字节：", [enc.decode_single_token_bytes(i) for i in ids])
    assert enc.decode(ids) == text  # decode 能完整还原
