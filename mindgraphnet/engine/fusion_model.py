"""
MindGraphNet - Hybrid Cross-Attention Fusion Neural Network
Fuses Transformer semantic embeddings, GAT structural embeddings, and psycholinguistic features
using cross-branch attention gating and temperature calibration.
"""
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import (
    TRANSFORMER_EMB_DIM, GRAPH_EMB_DIM, N_ENGINEERED_FEATURES,
    NUM_CLASSES, CLASS_NAMES
)

class MultiHybridFusionNet(nn.Module):
    def __init__(self, 
                 semantic_dim: int = TRANSFORMER_EMB_DIM, 
                 graph_dim: int = GRAPH_EMB_DIM, 
                 psych_dim: int = N_ENGINEERED_FEATURES, 
                 latent_dim: int = 64, 
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.graph_dim = graph_dim
        self.psych_dim = psych_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Modality projection heads into common latent subspace
        self.proj_semantic = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        self.proj_graph = nn.Sequential(
            nn.Linear(graph_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.proj_psych = nn.Sequential(
            nn.Linear(psych_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Cross-Modal Attention Gating: computes modality weights (semantic, graph, psych)
        self.modality_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        # Unified Fusion Classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes)
        )

        # Continuous risk score regressor head
        self.risk_regressor = nn.Sequential(
            nn.Linear(latent_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Learnable Temperature Scaling parameter for calibrated probabilities
        self.temperature = nn.Parameter(torch.ones(1) * 1.1)

    def forward(self, semantic_x: torch.Tensor, graph_x: torch.Tensor, 
                psych_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - calibrated_logits: [batch_size, num_classes]
            - continuous_risk_score: [batch_size, 1]
            - modality_weights: [batch_size, 3] (Semantic, Graph, Psycholinguistic)
        """
        h_sem = self.proj_semantic(semantic_x) # [B, latent_dim]
        h_gr = self.proj_graph(graph_x)       # [B, latent_dim]
        h_psy = self.proj_psych(psych_x)      # [B, latent_dim]

        # Concatenate projected modalities
        h_concat = torch.cat([h_sem, h_gr, h_psy], dim=-1) # [B, latent_dim * 3]

        # Compute dynamic attention gating weights per sample
        weights = self.modality_gate(h_concat) # [B, 3]
        w_sem = weights[:, 0:1]
        w_gr = weights[:, 1:2]
        w_psy = weights[:, 2:3]

        # Modulate representations
        h_fused = torch.cat([h_sem * w_sem, h_gr * w_gr, h_psy * w_psy], dim=-1)

        raw_logits = self.fusion_classifier(h_fused)
        calibrated_logits = raw_logits / torch.clamp(self.temperature, min=0.1)
        risk_score = self.risk_regressor(h_fused)

        return calibrated_logits, risk_score, weights
