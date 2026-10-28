"""推理篇 EP10 · Activation Function：ReLU vs GELU。

注意 GELU 有两个版本：
- 精确版：x · Φ(x)，Φ 是标准正态分布的累积分布函数。PyTorch 的 nn.GELU() 默认是这个。
- tanh 近似版：GPT-2 用的是这个，PyTorch 里要写 nn.GELU(approximate="tanh")。
两者差异很小，但要和官方权重逐位对齐，必须用 tanh 版。
"""
import _common  # noqa: F401
import torch
import torch.nn as nn

from gpt2.model import gelu

torch.manual_seed(42)
hidden = nn.Linear(4, 16)(torch.randn(3, 4))   # 「猫 吃 鱼」升维后，[3, 16]

relu_out = nn.ReLU()(hidden)
gelu_out = gelu(hidden)
print("ReLU 清零的维度数：", (relu_out == 0).sum().item(), "/", hidden.numel())
print("GELU 保留的负值数：", (gelu_out < 0).sum().item(), "/", hidden.numel())

x = torch.linspace(-4, 4, 1001)
print("\n精确版 vs tanh 近似版的最大差异：",
      (nn.GELU()(x) - nn.GELU(approximate="tanh")(x)).abs().max().item())
print("我们手写的 gelu 和 tanh 版的差异：",
      (gelu(x) - nn.GELU(approximate="tanh")(x)).abs().max().item())
