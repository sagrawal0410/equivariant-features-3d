import torch
import torch.nn as nn
from dataclasses import dataclass
import tyro



@dataclass
class BasicFCNConfig:
    in_dim: int
    out_dim: int
    h_dim: int = 128
    n_layers: int = 2
    use_bias: bool = True
    nonlinearity: str = 'relu'

    def setup(self) -> nn.Module:
        return BasicFCN(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            h_dim=self.h_dim,
            n_layers=self.n_layers,
            use_bias=self.use_bias,
            nonlinearity=self.nonlinearity
        )

class BasicFCN(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=128, n_layers=2, use_bias=True, nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = None
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'swish':
            self.nonlinearity = nn.SiLU()
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        if n_layers == 0:
            self.predict = nn.Identity()
        if n_layers == 1:
            self.predict = nn.Linear(in_dim, out_dim, bias=use_bias)
        if n_layers > 1:
            self.predict = [nn.Linear(in_dim, h_dim, bias=use_bias), ]
            self.predict = self.predict + ([nn.Linear(h_dim, h_dim, bias=use_bias)]*(n_layers-2))
            self.predict.append(nn.Linear(h_dim, out_dim, bias=use_bias))
            self.predict = nn.ModuleList(self.predict)

    def forward(self, x):
        x = self.predict[0](x)
        x = self.nonlinearity(x)
        for layer in self.predict[1:-1]:
            x = self.nonlinearity(layer(x))
        x = self.predict[-1](x)
        return x



@dataclass
class BasicCNNConfig:
    in_dim: int
    out_dim: int
    h_dim: int = 128
    n_layers: int = 3
    kernel_size: int = 3
    nonlinearity: str = 'relu'

    def setup(self) -> nn.Module:
        return BasicCNN(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            h_dim=self.h_dim,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            nonlinearity=self.nonlinearity
        )

class BasicCNN(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=128, n_layers=3, kernel_size=3, nonlinearity='relu'):
        super().__init__()
        self.nonlinearity = None
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'swish':
            self.nonlinearity = nn.SiLU()
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        if n_layers == 0:
            self.net = nn.Identity()
        if n_layers == 1:
            self.net = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2)
        if n_layers > 1:
            layers = []
            layers.append(nn.Conv2d(in_dim, h_dim, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(self.nonlinearity)
            for _ in range(n_layers - 2):
                layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel_size, padding=kernel_size//2))
                layers.append(self.nonlinearity)
            layers.append(nn.Conv2d(h_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2))
            self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


