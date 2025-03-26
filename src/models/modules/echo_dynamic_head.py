from torch import nn

from .basic_layers import MLP


class EchoDynamicEFHead(nn.Module):
    def __init__(self, cfg, in_features, bias_init=None):
        super().__init__()
        layers = cfg.model.head_layers
        self.net = MLP(in_features, [in_features]*(layers-1) + [1], nn.SiLU)
        if bias_init is not None:
            self.net.mlp[-1].bias.data[0] = bias_init

    def forward(self, x):
        x = self.net(x)
        return x


class EchoDynamicEFWithSigmoidHead(nn.Module):
    def __init__(self, cfg, in_features):
        super().__init__()
        layers = cfg.model.head_layers
        self.net = MLP(in_features, [in_features]*(layers-1) + [1], nn.SiLU)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        x = x * 100.0
        return x


class EchoDynamicClsHead(nn.Module):
    def __init__(self, cfg, in_features, num_classes=20):
        super().__init__()
        layers = cfg.model.head_layers
        assert num_classes >= 2
        if num_classes > 2:
            self.net = MLP(in_features, [in_features]*(layers-1) + [num_classes], nn.SiLU)
        else:  # binary classification
            self.net = MLP(in_features, [in_features]*(layers-1) + [1], nn.SiLU)

    def forward(self, x):
        x = self.net(x)
        return x