from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import FAULT_ORDER


FAULT_PREFIX_MAP = {
    'sd': 'sd_',
    'isc': 'isc_',
    'conn': 'conn_',
    'samp': 'samp_',
    'ins': 'ins_',
}


@dataclass
class FeatureGroupIndices:
    shared: list[int]
    fault_specific: dict[str, list[int]]


class FaultExpertHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.id_head = nn.Linear(hidden_dim, 1)
        self.warn_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.id_head(hidden), self.warn_head(hidden)


class MultiFaultDualTaskModel(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        feature_columns: np.ndarray,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_fault_specific_features: bool = True,
        use_expert_heads: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads for Transformer encoder.')

        self.feature_columns = [str(column) for column in feature_columns.tolist()]
        self.feature_groups = build_feature_group_indices(self.feature_columns)
        self.use_fault_specific_features = use_fault_specific_features
        self.use_expert_heads = use_expert_heads

        self.input_projection = nn.Linear(sequence_input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        if self.use_expert_heads:
            self.expert_heads = nn.ModuleDict()
            for fault in FAULT_ORDER:
                fault_feature_dim = len(self.feature_groups.fault_specific[fault]) if self.use_fault_specific_features else 0
                self.expert_heads[fault] = FaultExpertHead(hidden_dim + fault_feature_dim, hidden_dim, dropout)
        else:
            self.shared_id_head = nn.Linear(hidden_dim, len(FAULT_ORDER))
            self.shared_warn_head = nn.Linear(hidden_dim, len(FAULT_ORDER))

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = x_seq.size(1)
        encoded = self.input_projection(x_seq) + self.position_embedding[:, :seq_len, :]
        encoded = self.encoder(encoded)
        shared_repr = self.shared_projector(encoded.mean(dim=1))

        if not self.use_expert_heads:
            return {
                'id_logits': self.shared_id_head(shared_repr),
                'warn_logits': self.shared_warn_head(shared_repr),
            }

        id_logits = []
        warn_logits = []
        for fault in FAULT_ORDER:
            if self.use_fault_specific_features:
                fault_indices = self.feature_groups.fault_specific[fault]
                fault_features = x_feat[:, fault_indices] if fault_indices else x_feat.new_zeros((x_feat.size(0), 0))
            else:
                fault_features = x_feat.new_zeros((x_feat.size(0), 0))
            expert_input = torch.cat([shared_repr, fault_features], dim=-1)
            id_logit, warn_logit = self.expert_heads[fault](expert_input)
            id_logits.append(id_logit)
            warn_logits.append(warn_logit)

        return {
            'id_logits': torch.cat(id_logits, dim=1),
            'warn_logits': torch.cat(warn_logits, dim=1),
        }



def build_feature_group_indices(feature_columns: list[str]) -> FeatureGroupIndices:
    shared = [index for index, name in enumerate(feature_columns) if name.startswith('shared_')]
    fault_specific = {}
    for fault, prefix in FAULT_PREFIX_MAP.items():
        fault_specific[fault] = [index for index, name in enumerate(feature_columns) if name.startswith(prefix)]
    return FeatureGroupIndices(shared=shared, fault_specific=fault_specific)


class SequenceClassificationBaseline(nn.Module):
    def __init__(
        self,
        architecture: str,
        sequence_input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == 'lstm':
            self.encoder = nn.LSTM(
                input_size=sequence_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(hidden_dim, output_dim)
        elif architecture == 'transformer':
            if hidden_dim % num_heads != 0:
                raise ValueError('hidden_dim must be divisible by num_heads for transformer baseline.')
            self.input_projection = nn.Linear(sequence_input_dim, hidden_dim)
            self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f'Unsupported architecture: {architecture}')

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.architecture == 'lstm':
            _, (hidden, _) = self.encoder(x_seq)
            return self.classifier(hidden[-1])
        seq_len = x_seq.size(1)
        encoded = self.input_projection(x_seq) + self.position_embedding[:, :seq_len, :]
        encoded = self.encoder(encoded)
        return self.classifier(encoded.mean(dim=1))
