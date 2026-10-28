# 从零手写大模型 · 推理篇 14 - Text Generation（文本生成）

## 上集回顾

EP13 我们把 GPT-2 官方预训练权重加载进了自己手写的模型里,解决了 HuggingFace `Conv1D` 权重转置的经典坑。跑完那一集,我们拿到的其实只是**一次**前向传播的结果:给模型一句话,它吐出下一个最可能出现的 token id。

但这离"输入一句话,得到一段连贯续写"还差得远。这一集,我们把这最后一步补上——让模型真正能"说话"。

## 一次预测,不等于一段话

先把问题说清楚。EP13 里那行代码:

```python
next_token_id = logits[0, -1].argmax().item()
```

只回答了一个问题:"given 当前这句话,紧跟着的**下一个词**最可能是什么"。假设输入是 "the cat eats",这一步可能得到的答案是 "a"——就一个词,没有更多了。

如果你想要 "the cat eats a fish" 这种完整续写,思路其实很直观:**预测一个词,拼回去,再预测下一个,再拼,一直循环下去**。这个循环,就是这一集要写的东西。

## 自回归生成(Autoregressive Generation)

这套"预测→拼接→再预测"的机制有个专门的名字:**自回归生成**。之所以叫"自回归",是因为模型每一步的输入,都包含了它自己前面所有步骤生成出来的内容——用自己的输出,喂给自己做下一次输入。

伪代码写出来是这样:

```python
def generate(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(input_ids)                      # 前向传播
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)  # [batch, 1]
        input_ids = torch.cat([input_ids, next_id], dim=1)  # 拼接到序列末尾
    return input_ids
```

每循环一次,序列长度 +1。跑 `max_new_tokens` 次,就能拿到一整段续写。用"the cat eats"举例,循环过程大概长这样:

```
第1步: "the cat eats"          → 预测 "a"    → 拼接 "the cat eats a"
第2步: "the cat eats a"        → 预测 "fish" → 拼接 "the cat eats a fish"
第3步: "the cat eats a fish"   → 预测 "."    → 拼接 "the cat eats a fish."
```

看起来简单,但这里藏着一个容易被忽略的效率问题。

## 一个值得注意的细节:每一步都在重复计算

注意上面伪代码里,**每一次循环都把完整序列重新喂给模型跑一遍前向传播**。第 1 步算了 3 个 token 的 attention,第 2 步算了 4 个 token 的,第 3 步算了 5 个——前面已经算过的部分(比如 "the cat eats" 这几个词的 K、V),其实每次都在被重复计算。

这个问题工业界有个标准解法,叫 **KV Cache**(把已经算好的 Key/Value 缓存下来,新的一步只用算新增的那个 token,不用从头重算),是推理加速的核心机制之一。但这一集我们先不引入这个优化——目的是把"自回归生成"这个核心逻辑讲透,KV Cache 属于"性能优化"范畴,值得单独开一集来讲,这里先用最朴素的"每次重新算全部"版本把流程跑通。

## Greedy Decoding 的问题:太死板

上面这种"每次都选概率最大的那个词"的策略,叫 **Greedy Decoding(贪心解码)**。它的问题是:**只要有一步选错(或者说,选择了一个"看起来对、但不是最优"的词),后面所有的生成都会被这一步锁死**,因为没有回头路——模型不会"反悔"已经生成出来的词。

而且贪心解码还有个更直观的毛病:**容易生成重复、乏味的文本**。比如遇到某些上下文,模型可能反复吐出同一个词或同一个短语,因为"概率最大"这个策略每次都做同样的选择,没有给"不那么确定但也合理"的候选词任何机会。

这就是为什么几乎所有实际的语言模型服务(包括 ChatGPT 这类产品)都不会单纯用贪心解码,而是引入**随机性**。

## 采样(Sampling):引入随机性

采样的思路很简单:不再死板地选"概率最大"的那个词,而是把 logits 转换成一个概率分布(用 Softmax),然后**按这个概率分布随机抽一个词**——概率高的词被抽中的机会更大,但概率稍低的词也有机会被选中。

```python
import torch.nn.functional as F

probs = F.softmax(logits[0, -1], dim=-1)   # 转成概率分布
next_id = torch.multinomial(probs, num_samples=1)  # 按概率随机抽样
```

这样一来,同一句话输入两次,可能得到两种不同的续写——这也是为什么你跟 ChatGPT 这类产品对话,同样的问题问两次,答案经常不完全一样的原因。

## Temperature:控制"随机的程度"

纯采样也有问题:如果概率分布比较"平"(很多词概率都差不多),完全随机抽样可能抽出一个语法都不通的词。**Temperature** 就是用来调节这个"随机程度"的旋钮。

做法是在 Softmax 之前,把 logits 先除以一个数 `T`(temperature):

```python
probs = F.softmax(logits[0, -1] / temperature, dim=-1)
```

- **`T < 1`(比如 0.7)**:相当于把 logits 之间的差距拉大,概率分布变得更"尖锐"——高概率的词更容易被选中,生成结果更保守、更接近贪心解码,但也更连贯、更不容易跑偏
- **`T = 1`**:不做任何调整,就是原始的概率分布
- **`T > 1`(比如 1.5)**:把 logits 之间的差距压平,概率分布变得更"均匀"——低概率的词也有更大机会被选中,生成结果更有创意、更多样,但也更容易语无伦次

**极限情况**:`T → 0` 时,几乎等价于贪心解码(概率最大的词会以接近 100% 的概率被选中);`T` 非常大时,几乎等价于纯随机(每个词被选中的概率趋于均等)。

## Top-k:只在"靠谱候选"里随机

Temperature 解决了"随机程度"的问题,但还有一个风险:哪怕经过 temperature 调整,词表里 5 万多个词,理论上任何一个词都有非零的概率被抽中——包括那些完全不合理的词。**Top-k** 就是用来兜底这个问题的。

做法是:在做 Softmax 和采样之前,**先把概率最高的 k 个词挑出来,其余的全部丢弃(概率设为 0)**,只在这 k 个"靠谱候选"里做随机采样。

```python
def top_k_filter(logits, k):
    values, indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[indices] = values
    return filtered_logits

logits = top_k_filter(logits[0, -1], k=50)
probs = F.softmax(logits / temperature, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
```

比如 `k=50`,意味着不管词表有多大,每一步只在概率最高的 50 个词里随机选——既保留了随机性带来的多样性,又避免了抽到完全离谱的词。GPT-2 官方生成时常用的默认值就是 `k=50` 左右。

## 什么时候该停:EOS Token

前面讲的都是"怎么生成下一个词",还有一个问题没解决:**这个循环什么时候该停?**

如果不设停止条件,`generate` 函数会一直跑到你设定的 `max_new_tokens` 才停,哪怕模型其实早就想"说完了"。解决办法是引入 **EOS(End of Sequence)token**——训练语料里,每段文本结束的地方都会标一个特殊的 token(GPT-2 里是 `<|endoftext|>`,id 是 50256)。只要生成过程中采样到了这个 token,就说明模型认为"这段话该结束了",直接停止循环。

```python
def generate(model, input_ids, max_new_tokens, eos_token_id=50256):
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_id = sample_next_token(logits)  # 前面讲的采样逻辑
        if next_id.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
```

## 完整拼装

把上面几块拼在一起,就是一个相对完整的生成函数:

```python
def generate(model, input_ids, max_new_tokens, temperature=0.8, top_k=50, eos_token_id=50256):
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[0, -1] / temperature
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    return input_ids
```

拿 "the cat eats" 跑一遍,这次不再只拿到一个 token id,而是能解码出一整句连贯的续写——比如 "the cat eats a fish and falls asleep"这样的结果(具体内容取决于采样的随机性,每次跑可能不完全一样)。

## 收尾:从 EP01 到 EP14,我们做了什么

回头看一下这条走了 14 集的路:从 Tokenizer 把文字切成 token,到 Embedding 把 token 变成向量,到位置编码让模型知道词的顺序,到 Self-Attention 和 Multi-Head Attention 让模型学会"关注"上下文里的相关部分,到残差连接和 LayerNorm 让深层网络能稳定训练,到 FFN 和激活函数给模型非线性表达能力,到把这些拼成完整的 Transformer Block 并堆叠成深层网络,再到加载 GPT-2 真实预训练权重,最后到这一集的文本生成——**一个完全由你自己手写、参数是真实的、能进行自回归对话的 GPT-2,到这里就跑通了。**

这也是这个系列最初立下的目标:"Level 1"——不训练,但把从 Tokenizer 到推理的每一步都亲手实现一遍,把黑盒拆开看清楚里面每一层到底在算什么。到 EP14,这个目标完整达成了。

## 后记

关于"训练"——也就是怎么让一个随机初始化的模型,从零学会你想要的规律——这是完全不同的另一套机制(损失函数、反向传播、优化器、训练循环),不在这个系列最初划定的范围里。感兴趣的话,这会是之后一个新阶段的内容。

文本生成这一环到这里告一段落。下一期（第 15 期）是系列完结篇，我们会聊聊可解释性：已经知道模型怎么算之后，怎样进一步理解它内部的表示和计算。
