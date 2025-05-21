import torch
import torch.nn as nn


class RevInMultivariate(nn.Module):
    """
    ReversibleInstanceNorm1d for Multivariate models
    """
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize ReVIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def _normalize(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=1, keepdim=True).detach()
        self.batch_std = torch.sqrt(
            torch.var(x, axis=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # Reverse the normalization
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight

        x = x * self.batch_std
        x = x + self.batch_mean
        return x
