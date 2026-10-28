"""推理篇 EP07 · Residual Connection：输出 = 输入 + 子层(输入)。"""
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(3, 8)            # 「猫 吃 鱼」，[3, 8]
attn_output = torch.randn(3, 8)  # 假设这是注意力层的输出，形状必须和 X 一致

output = X + attn_output         # 残差连接就是一次加法
print(X.shape, attn_output.shape, output.shape)


class Residual(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(x)


# 直观感受：哪怕子层输出全是 0，信息也能原样传下去
zero_layer = Residual(nn.Linear(8, 8))
nn.init.zeros_(zero_layer.sublayer.weight)
nn.init.zeros_(zero_layer.sublayer.bias)
print("子层输出为 0 时，输出 == 输入：", torch.equal(zero_layer(X), X))
