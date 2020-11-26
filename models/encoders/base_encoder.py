import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

        self.in_channels = 3
        self.out_channels = (0, 0, 0, 0, 0)

    def forward(self, x, get_pyramid=True):
        x0 = self.init_block(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if get_pyramid:
            return [x0, x1, x2, x3, x4]
        else:
            return x4

    def get_out_channels(self, mock_tensor):
        xs = self.forward(mock_tensor, get_pyramid=True)
        return [x.shape[1] for x in xs]

    def get_out_channels_2d(self):
        mock_tenor = torch.zeros((1, 3, 64, 64))
        return self.get_out_channels(mock_tenor)

    def get_out_channels_3d(self):
        mock_tenor = torch.zeros((1, 3, 64, 64, 64))
        return self.get_out_channels(mock_tenor)
