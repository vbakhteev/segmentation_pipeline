import torch
from torch import nn

from models.utils import get_layers_by_dim


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
        self.eval()
        mock_tenor = torch.zeros((1, self.in_channels, 128, 128))
        return self.get_out_channels(mock_tenor)

    def get_out_channels_3d(self):
        mock_tenor = torch.zeros((1, self.in_channels, 128, 128, 128))
        return self.get_out_channels(mock_tenor)

    def change_in_channels(self, num_channels: int, n_dim: int):
        if self.in_channels == num_channels:
            return

        named_conv = next(self.named_parameters())
        name = named_conv[0].split(".")[:-1]
        w = named_conv[1]  # (out_filters, 3, k, k) or (out_filters, 3, k, k, k)

        nn_module = self
        for n in name[:-1]:
            nn_module = getattr(nn_module, n)
        first_conv = next(nn_module.children())

        Conv = get_layers_by_dim(n_dim)["conv"]
        setattr(
            nn_module,
            name[-1],
            Conv(
                num_channels,
                w.shape[0],
                kernel_size=w.shape[2:],
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
            ),
        )

        if num_channels < self.in_channels:
            w = w[:, :num_channels, ...]
            getattr(nn_module, name[-1]).weight = nn.Parameter(w)

        elif num_channels > self.in_channels:
            shape = list(w.shape)
            shape[1] = num_channels

            new_w = torch.empty(shape)
            nn.init.xavier_normal_(new_w)
            new_w[:, :num_channels, ...] = w

            getattr(nn_module, name[-1]).weight = nn.Parameter(new_w)

        self.in_channels = num_channels
