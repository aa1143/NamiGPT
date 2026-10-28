# 从零手写大模型 · 推理篇 13 - Loading Pretrained Weights（加载预训练权重）

## 上集回顾

EP12 我们把 Transformer Block 一层层堆了起来,拼出了完整的 GPT 前向传播:embedding → N 层 Transformer Block → 最终 LayerNorm → 输出头。整个模型的"骨架"已经搭完了。

但骨架里的参数,到目前为止全是 `torch.randn` 随机初始化出来的。这意味着——之前每一集我们反复强调的那句免责声明依然成立:**这个模型现在跑起来,输出没有任何语义意义。**

这一集,我们要给这具骨架填上血肉:加载 GPT-2 官方发布的预训练权重。

## 为什么这一步不是"随便赋值"

有人可能会想:不就是把训练好的数字塞进对应的矩阵里吗,能有多难?

难点不在"塞",而在"对齐"。你手写的 `GPT` 类里,每一层参数的命名方式、每个矩阵的存储方向,都是你自己定的。而 HuggingFace 官方发布的 GPT-2 权重,是按照 OpenAI 原始 TensorFlow 实现的习惯存下来的。两边的"方言"不一样,直接赋值大概率会报 shape 不匹配,或者更隐蔽的——shape 凑巧对上了,但数值全错,模型输出一堆乱码而你还以为自己代码写对了。

这一集要解决的核心问题就是:**把 HuggingFace 的权重字典,翻译成你自己模型认识的样子。**

## 一个绕不开的现实:猫吃鱼要先放一放

前面十几集我们一直用"猫吃鱼"这个 `d_model=4` 的三词序列做手算示例,因为参数少、能一步步手推。但预训练权重是别人在真实语料上训出来的,维度是固定的——GPT-2 最小的版本(gpt2-small)配置是:

```
vocab_size = 50257
n_embd     = 768   # 也就是我们之前说的 d_model
n_layer    = 12
n_head     = 12
n_ctx      = 1024  # 最大序列长度
```

也就是说,这一集开始,我们的 `d_model` 从手算用的 4 变成了真实的 768。"猫吃鱼"这个例子还会用,但只用来验证"流程走没走通",不再逐数字手推——768 维的向量没法在文章里摆开手算,这个转变是必然的,不代表前面几集的示例白做了,理解 shape 变化的原理比记住具体数字更重要。

## 权重从哪来:HuggingFace 的 state_dict

我们不需要自己下载 OpenAI 的原始 TensorFlow checkpoint 再手动解析,`transformers` 库已经把 GPT-2 的权重转成了 PyTorch 格式,直接加载就能拿到一个 `state_dict`(本质是一个 `参数名 -> Tensor` 的字典):

```python
from transformers import GPT2LMHeadModel

hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_state_dict = hf_model.state_dict()

for name, tensor in list(hf_state_dict.items())[:5]:
    print(name, tensor.shape)
```

输出大概长这样:

```
transformer.wte.weight        torch.Size([50257, 768])
transformer.wpe.weight        torch.Size([1024, 768])
transformer.h.0.ln_1.weight   torch.Size([768])
transformer.h.0.ln_1.bias     torch.Size([768])
transformer.h.0.attn.c_attn.weight  torch.Size([768, 2304])
```

前两行好认:`wte` 是 token embedding,`wpe` 是位置嵌入（第 04 期介绍位置编码，第 12 期改用了可学习的位置嵌入）。`h.0` 就是第 0 层 Transformer Block。到这里都还算顺利。

## 核心坑:Conv1D 不是 nn.Linear

问题出在 `attn.c_attn.weight` 这一行。我们在 EP05(Self-Attention)里,Q/K/V 的投影用的是 `nn.Linear`,它的权重 shape 是 `[out_features, in_features]`。

但你仔细看上面那行输出:`c_attn.weight` 的 shape 是 `[768, 2304]`——如果按 `nn.Linear` 的约定去理解,这应该是"输入 768,输出 2304"没错(2304 = 768×3,因为 Q/K/V 一次性投影),看起来对得上。

坑就藏在这:**这个矩阵实际上是转置过的。**

原因是 OpenAI 原始实现用的不是 PyTorch 的 `nn.Linear`,而是 TensorFlow 风格的 `Conv1D`(在 `transformers` 库里叫 `Conv1D`,和卷积没关系,纯粹是历史命名)。`Conv1D` 做矩阵乘法时用的是 `x @ W`,权重 shape 是 `[in_features, out_features]`;而 `nn.Linear` 做的是 `x @ W.T`,权重 shape 是 `[out_features, in_features]`。

两者的计算结果可以等价,但**存储方向是反的**。如果你把 `Conv1D` 存的权重原封不动塞进 `nn.Linear`,shape 检查可能都过不了(1024×768 塞不进 768×1024 的 `nn.Linear`,会直接报错——这种情况还算幸运);更麻烦的是某些方阵形状凑巧对称,shape 检查完全通过,但矩阵乘法方向反了,模型输出全是乱码,你还得反过来排查半天。

**解决办法就一句话:凡是从 `c_attn`、`c_proj`(attention 和 FFN 里的那几个)读出来的权重,加载前先转置一次。**

```python
# 错误:直接赋值,方向反了
my_model.attn.qkv_proj.weight.data = hf_state_dict["transformer.h.0.attn.c_attn.weight"]

# 正确:先转置,再赋值
my_model.attn.qkv_proj.weight.data = hf_state_dict["transformer.h.0.attn.c_attn.weight"].T
```

这个坑几乎是每个从零复现 GPT-2 的人都会踩一次的地方,记住一句话就够了:**看到 `c_attn` 或 `c_proj`,先 `.T` 再说。** LayerNorm 和 embedding 的权重不涉及这个问题,原样赋值即可。

## 参数名对照表

把 HuggingFace 的命名和我们自己模型里的命名一一对上:

| HuggingFace 权重名 | 对应我们模型里的 | 需要转置? |
|---|---|---|
| `transformer.wte.weight` | `token_embedding.weight` | 否 |
| `transformer.wpe.weight` | `position_embedding.weight` | 否 |
| `transformer.h.{i}.ln_1.weight/bias` | 第 i 层 Block 的 `ln1` | 否 |
| `transformer.h.{i}.attn.c_attn.weight/bias` | 第 i 层的 Q/K/V 合并投影 | 是 |
| `transformer.h.{i}.attn.c_proj.weight/bias` | 第 i 层 attention 的输出投影 | 是 |
| `transformer.h.{i}.ln_2.weight/bias` | 第 i 层 Block 的 `ln2` | 否 |
| `transformer.h.{i}.mlp.c_fc.weight/bias` | 第 i 层 FFN 的升维层 | 是 |
| `transformer.h.{i}.mlp.c_proj.weight/bias` | 第 i 层 FFN 的降维层 | 是 |
| `transformer.ln_f.weight/bias` | 最终的 `final_ln` | 否 |

## 完整加载代码

```python
def load_gpt2_weights(my_model, hf_state_dict, n_layer=12):
    sd = hf_state_dict

    # embedding 层,原样赋值
    my_model.token_embedding.weight.data = sd["transformer.wte.weight"]
    my_model.position_embedding.weight.data = sd["transformer.wpe.weight"]

    for i in range(n_layer):
        block = my_model.blocks[i]
        prefix = f"transformer.h.{i}."

        # LayerNorm:原样赋值
        block.ln1.weight.data = sd[prefix + "ln_1.weight"]
        block.ln1.bias.data   = sd[prefix + "ln_1.bias"]
        block.ln2.weight.data = sd[prefix + "ln_2.weight"]
        block.ln2.bias.data   = sd[prefix + "ln_2.bias"]

        # Attention:c_attn / c_proj 都要转置
        block.attn.qkv_proj.weight.data = sd[prefix + "attn.c_attn.weight"].T
        block.attn.qkv_proj.bias.data   = sd[prefix + "attn.c_attn.bias"]
        block.attn.out_proj.weight.data = sd[prefix + "attn.c_proj.weight"].T
        block.attn.out_proj.bias.data   = sd[prefix + "attn.c_proj.bias"]

        # FFN:c_fc / c_proj 都要转置
        block.ffn.fc1.weight.data = sd[prefix + "mlp.c_fc.weight"].T
        block.ffn.fc1.bias.data   = sd[prefix + "mlp.c_fc.bias"]
        block.ffn.fc2.weight.data = sd[prefix + "mlp.c_proj.weight"].T
        block.ffn.fc2.bias.data   = sd[prefix + "mlp.c_proj.bias"]

    # 最终 LayerNorm
    my_model.final_ln.weight.data = sd["transformer.ln_f.weight"]
    my_model.final_ln.bias.data   = sd["transformer.ln_f.bias"]

    # 输出头:GPT-2 的输出头和 token embedding 权重共享(weight tying)
    my_model.output_head.weight.data = sd["transformer.wte.weight"]

    return my_model
```

最后一行藏了一个额外的知识点:GPT-2 的输出头(把 768 维向量映射回 50257 维词表 logits 的那一层)和输入的 token embedding 是**共享同一份权重**的,这叫 weight tying,能省下一大块参数量(50257×768 ≈ 3860 万参数)。这个点值得单独记一下,不然你会发现 `hf_state_dict` 里根本找不到一个叫 `lm_head.weight` 的独立参数——不是漏了,是它和 `wte.weight` 是同一个东西。

## 验证:随机权重 vs 预训练权重

加载完之后,用"猫吃鱼"对应的英文短句跑一遍前向传播,对比效果最直观。之前用随机权重时,取 logits 最大值对应的 token,拼出来的下一个词基本是乱码;换上预训练权重后,至少能看到语法通顺、语义相关的续写倾向——这就是这一集要验证的东西:**参数从"随机数字"变成了"从海量文本里学出来的统计规律"。**

```python
import torch

my_model = GPT(config)  # EP12 里写的模型
my_model = load_gpt2_weights(my_model, hf_state_dict)
my_model.eval()

input_ids = torch.tensor([[464, 3797, 15070]])  # 编码后的 "The cat eats" 之类的示例
with torch.no_grad():
    logits = my_model(input_ids)

next_token_id = logits[0, -1].argmax().item()
print(next_token_id)  # 现在这个预测才第一次"有意义"
```

这里先只看能不能跑通、输出是否合理,完整的"怎么把 token id 序列变成人话续写"——也就是文本生成的完整逻辑(贪心 vs 采样、temperature、top_k、什么时候停下来),留到下一集专门展开。

## 下集预告

EP14 我们正式进入推理阶段:文本生成(text generation)。会讲清楚 greedy decoding 和采样(sampling)的区别、temperature 和 top_k 怎么控制生成的"随机程度"、以及 EOS token 怎么让模型知道该停下了。到那一集结束,你手写的这个模型就能真正"对话"起来了。
