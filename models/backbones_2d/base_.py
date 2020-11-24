import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, x, get_pyramid=False):
        x0 = self.init_block(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if get_pyramid:
            return [x0, x1, x2, x3, x4]
        else:
            return x4

    def get_out_channels(self):
        mock = torch.zeros((1, 3, 64, 64))
        with torch.no_grad():
            out = self.forward(mock)

        return out.shape[1]
