import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cva_net.alexnet.backbone.model import AlexNetBackbone


@dataclass
class Config:
    latent_dim: int = 128
    ema_tau_min: float = 0.996
    ema_tau_max: float = 0.999
    ema_total_steps: int = 1000


class JEPA(nn.Module):

    def __init__(
        self,
        backbone: AlexNetBackbone,
        latent_dim: int=128,
        ema_tau_min=0.996,
        ema_tau_max=0.999,
        ema_total_steps=1000,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_tau_max = ema_tau_max
        self.ema_tau_min = ema_tau_min
        self.ema_total_steps = ema_total_steps
        self.ema_steps = 0

        # Context encoder (online)
        self.context_encoder = backbone

        # Target encoder (EMA du context encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def _get_ema_tau(self):
        """Calcul cosinus pour le tau de l'EMA"""
        if self.ema_steps >= self.ema_total_steps:
            return self.ema_tau_min
        cos_schedule = 0.5 * (1 + np.cos(np.pi * self.ema_steps / self.ema_total_steps))
        self.ema_steps += 1
        return self.ema_tau_min + (self.ema_tau_max - self.ema_tau_min) * cos_schedule

    def update_target_encoder(self):
        """Mise à jour EMA du target encoder"""
        tau = self._get_ema_tau()
        with torch.no_grad():
            for online_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                target_param.data = tau * target_param.data + (1 - tau) * online_param.data

    def forward(self, context_view, target_view):
        # Encoder les vues.
        context_emb = self.context_encoder(context_view)

        with torch.no_grad():
            target_emb = self.target_encoder(target_view)

        # Prédire la représentation target depuis context.
        predicted_target = self.predictor(context_emb)

        return predicted_target, target_emb, context_emb
