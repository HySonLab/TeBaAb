import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple


def get_activation(type: str, dim: int) -> nn.Module:
    if type == "relu":
        return nn.ReLU()
    elif type == "leaky_relu":
        return nn.LeakyReLU()
    elif type == "elu":
        return nn.ELU()
    elif type == "prelu":
        return nn.PReLU(dim)
    else:
        raise ValueError(f"Activation {type} is not supported.")


class Deconv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: Union[Tuple[int, int], int] = 2,
                 padding: int = 0,
                 dropout: float = 0.2,
                 activation_type: str = "relu"):
        super(Deconv1d, self).__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.deconv = nn.ConvTranspose1d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = get_activation(activation_type, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Defined Deconvolution operation

        Args:
            x (Tensor): tensor of shape `[batch, in_channels, dim1]`

        Returns:
            Tensor: tensor of shape `[batch, out_channels, dim2]`
        """
        x = self.dropout(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Upsampling(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 max_filters: int,
                 low_res_dim: int = 64,
                 min_deconv_dim: int = 32,
                 num_deconv_layers: int = 3,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 0,
                 dropout: float = 0.2,
                 activation_type: str = "relu",
                 device: Union[torch.device, str] = "cpu"):
        super(Upsampling, self).__init__()
        self.low_res_dim = low_res_dim
        low_res_features = min(min_deconv_dim * 2 ** num_deconv_layers, max_filters)
        self.low_res_features = low_res_features
        self.latent_lin = nn.Linear(latent_dim, low_res_dim * low_res_features)
        self.deconvs = nn.ModuleList()
        for i in range(num_deconv_layers):
            out_channels = min(min_deconv_dim * 2 ** (num_deconv_layers - i - 1), max_filters)
            deconv = Deconv1d(low_res_features, out_channels, kernel_size,
                              stride, padding, dropout, activation_type)
            self.deconvs.append(deconv)
            low_res_features = out_channels

    def forward(self, latent: Tensor) -> Tensor:
        """Upsampling

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`

        Returns:
            h (Tensor):  tensor of shape `[B, min_deconv_dim, low_res_dim * stride**n_deconv_layer]`
        """
        h = self.latent_lin(latent)
        h = h.view(-1, self.low_res_features, self.low_res_dim)
        for deconv_layer in self.deconvs:
            h = deconv_layer(h)

        h = h.transpose(1, 2)

        return h


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super(LatentEncoder, self).__init__()
        self.linear_mu = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim * 2, latent_dim))
        self.linear_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

    def forward(self, hidden: Tensor):
        mu = self.linear_mu(hidden)
        logsigma = self.linear_logsigma(hidden)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(logsigma * 0.5)
        return z, mu, logsigma


class KLDivergence:
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def __call__(self, means: Tensor, logvars: Tensor) -> Tensor:
        kl_cost = -0.5 * (1.0 + logvars - means ** 2 - logvars.exp())
        kl_cost = torch.sum(kl_cost, 1)
        if self.reduction == "none":
            return kl_cost
        elif self.reduction == "sum":
            return torch.sum(kl_cost)
        else:
            return torch.mean(kl_cost, 0)
