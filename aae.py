import torch.nn as nn
import torch

m = nn.Conv2d(3, 10, kernel_size=1, stride=1)
x = torch.randn(6, 3, 7, 7)

print(m(x).size())