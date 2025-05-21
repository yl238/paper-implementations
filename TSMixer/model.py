import torch.nn as nn
import torch.nn.functional as F
from utils import RevInMultivariate


class TemporalMixing(nn.Module):
    """Temporal mixing"""

    def __init__(self, n_series, input_size, dropout):
        super().__init__()
        self.temporal_norm = nn.BatchNorm1d(
            num_features=n_series * input_size, eps=0.001, momentum=0.01
        )
        self.temporal_lin = nn.Linear(input_size, input_size)
        self.temporal_drop = nn.Dropout(dropout)

    def forward(self, input):
        # Get shapes
        batch_size = input.shape[0]
        input_size = input.shape[1]
        n_series = input.shape[2]

        # Temporal MLP
        x = input.permute(0, 2, 1)  # [B, L, N] -> [B, N, L]
        x = x.reshape(batch_size, -1)  # [B, N, L] -> [B, N * L]
        x = self.temporal_norm(x)  # [B, N * L] -> [B, N * L]
        x = x.reshape(batch_size, n_series, input_size)  # [B, N * L] -> [B, N, L]
        x = F.relu(self.temporal_lin(x))  # [B, N, L] -> [B, N, L]
        x = x.permute(0, 2, 1)  # [B, N, L] -> [B, L, N]
        x = self.temporal_drop(x)  # [B, L, N] -> [B, L, N]

        return x + input  # Residual


class FeatureMixing(nn.Module):
    """FeatureMixing"""

    def __init__(self, n_series, input_size, dropout, ff_dim):
        super().__init__()
        self.feature_norm = nn.BatchNorm1d(
            num_features=n_series * input_size, eps=0.001, momentum=0.01
        )
        self.feature_lin_1 = nn.Linear(n_series, ff_dim)
        self.feature_lin_2 = nn.Linear(ff_dim, n_series)
        self.feature_drop_1 = nn.Dropout(dropout)
        self.feature_drop_2 = nn.Dropout(dropout)

    def forward(self, input):
        # Get shapes
        batch_size = input.shape[0]
        input_size = input.shape[1]
        n_series = input.shape[2]

        # Feature MLP
        x = input.reshape(batch_size, -1)  # [B, L, N] -> [B, L * N]
        x = self.feature_norm(x)  # [B, L * N] -> [B, L * N]
        x = x.reshape(batch_size, input_size, n_series)  # [B, L * N] -> [B, L, N]
        x = F.relu(self.feature_lin_1(x))  # [B, L, N] -> [B, L, ff_dim]
        x = self.feature_drop_1(x)  # [B, L, ff_dim] -> [B, L, ff_dim]
        x = self.feature_lin_2(x)  # [B, L, ff_dim] -> [B, L, N]
        x = self.feature_drop_2(x)  # [B, L, N] -> [B, L, N]

        return x + input  # Residual


class MixingLayer(nn.Module):
    """
    MixingLayer
    """

    def __init__(self, n_series, input_size, dropout, ff_dim):
        super().__init__()
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(n_series, input_size, dropout)
        self.feature_mixer = FeatureMixing(n_series, input_size, dropout, ff_dim)

    def forward(self, input):
        x = self.temporal_mixer(input)
        x = self.feature_mixer(x)
        return x


class TemporalProjection(nn.Module):
    """
    Temporal projection
    """

    def __init__(self, input_size, forecast_horizon):
        super().__init__()
        self.lin = nn.Linear(in_features=input_size, out_features=forecast_horizon)

    def forward(self, input):
        # Input is [B, L, N]
        x = input.permute(0, 2, 1)  # [B, N, L]
        x = self.lin(x)
        x = x.permute(0, 2, 1)  # [B, T, N]
        return x


class TSMixer(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        n_series,
        loss,
        n_block=2,
        revin=True,
        ff_dim=64,
        dropout=0.9,
        batch_size: int = 32,
        learning_rate: float = 1.0e-4,
        step_size: int = 1,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            batch_size=batch_size,
            n_block=n_block,
            loss=loss,
            revin=revin,
            learning_rate=learning_rate,
            step_size=step_size,
        )
        # Reversible InstanceNormalization layer
        self.revin = revin
        if self.revin:
            self.norm = RevInMultivariate(num_features=n_series, affine=True)

        # Mixing layers
        mixing_layers = [
            MixingLayer(
                n_series=n_series,
                input_size=input_size,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            for _ in range(n_block)
        ]
        self.mixing_layers = nn.Sequential(*mixing_layers)

        # Temporal projection
        self.temporal_projection = TemporalProjection(
            input_size=input_size, forecast_horizon=h
        )

    def forward(self, window_batch):
        # Parse batch
        x = window_batch["insample_y"]  # x: [batch_size, input_size, n_series]
        batch_size = x.shape[0]

        # TSMixer: InstanceNorm + Mixing layers + Dense output layer + ReverseInstanceNorm
        if self.revin:
            x = self.norm(x, "norm")
        x = self.mixing_layers(x)
        x = self.temporal_projection(x)
        if self.revin:
            x = self.norm(x, "denorm")

        x = x.reshape(
            batch_size, self.h, self.n_series
        )
        return x
