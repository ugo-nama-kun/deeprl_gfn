import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """
    Random shifting augmentation block from DrQ-v2
    https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad):
        super(RandomShiftsAug, self).__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


# resnet block from: https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualLayer(nn.Module):
    """ Single Residual Layer """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualLayer, self).__init__()
        self.conv = conv3x3(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out


# Orthogonal Initialization of Weights
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class BetaHead(nn.Module):

    def forward(self, c1, c0):
        return torch.distributions.Independent(
            torch.distributions.Beta(c1, c0), 1
        )


class BetaPolicyModel(nn.Module):
    def __init__(self, action_size, obs_size, hidden1, hidden2=None, activation=nn.Tanh):
        super(BetaPolicyModel, self).__init__()

        if hidden2:
            self.fc = torch.nn.Sequential(
                nn.Linear(obs_size, hidden1),
                activation(),
                nn.Linear(hidden1, hidden2),
                activation()
            )
            ortho_init(self.fc[0], gain=1)
            ortho_init(self.fc[2], gain=1)

            final_hidden = hidden2
        else:
            self.fc = torch.nn.Sequential(
                nn.Linear(obs_size, hidden1),
                activation(),
            )
            ortho_init(self.fc[0], gain=1)

            final_hidden = hidden1

        self.fcc_c0 = nn.Linear(final_hidden, action_size)
        nn.init.orthogonal_(self.fcc_c0.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c0.bias)

        self.fcc_c1 = nn.Linear(final_hidden, action_size)
        nn.init.orthogonal_(self.fcc_c1.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c1.bias)

        self.beta_head = BetaHead()

    def forward(self, x):
        feature = self.fc(x)
        c0 = torch.log(1 + torch.exp(self.fcc_c0(feature))) + 1
        c1 = torch.log(1 + torch.exp(self.fcc_c1(feature))) + 1
        return self.beta_head(c0, c1)
