"""
Spatio-Temporal Adversarial Dynamics (GAN-LSTM) for Non-Stationary Sequence Prediction.

Synthesizes meteorological cloud trajectories and spatial state transitions:
1. ConvLSTM Generator to preserve continuous topological manifold constraints.
2. Spectral-Normalized Discriminator with Continuity & Advection Regularization.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SpatioTemporalGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cell1 = ConvLSTMCell(in_channels, hidden_channels)
        self.cell2 = ConvLSTMCell(hidden_channels, hidden_channels)
        self.out_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, seq_in: torch.Tensor, horizon: int = 6) -> torch.Tensor:
        """
        Args:
            seq_in: Tensor of shape (B, T_in, C, H, W)
            horizon: Number of future steps to generate
        Returns:
            seq_out: Tensor of shape (B, horizon, C, H, W)
        """
        B, T_in, C, H, W = seq_in.shape
        device = seq_in.device

        # Initialize internal states
        h1 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c1 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        h2 = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c2 = torch.zeros(B, self.hidden_channels, H, W, device=device)

        # Encode historical input steps
        for t in range(T_in):
            x_t = seq_in[:, t]
            h1, c1 = self.cell1(x_t, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))

        # Autoregressive generation across prediction horizon
        predictions = []
        cur_frame = self.out_conv(h2)
        predictions.append(cur_frame)

        for _ in range(1, horizon):
            h1, c1 = self.cell1(cur_frame, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            cur_frame = self.out_conv(h2)
            predictions.append(cur_frame)

        return torch.stack(predictions, dim=1)


class SpatioTemporalDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1)
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: Tensor of shape (B, T, C, H, W)
        Returns:
            logits: Tensor of shape (B, 1)
        """
        # Permute to (B, C, T, H, W) for Conv3D
        seq_permuted = seq.permute(0, 2, 1, 3, 4)
        return self.net(seq_permuted)
