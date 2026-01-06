"""
Neural network backbones for processing GW strain data and physics features.

Provides:
- StrainCNN: 1D CNN for raw strain time series
- PhysicsMLP: MLP for auxiliary physics features
- HybridBackbone: Fusion of both streams
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class StrainCNN(nn.Module):
    """
    1D CNN for extracting features from strain time series.

    Architecture designed for moderate-length signals (~2048-4096 samples at 1024 Hz).
    Uses aggressive striding to reduce sequence length while preserving temporal structure.

    Adapted from Maxime's SkyLocalizationCNN.
    """

    def __init__(
        self,
        in_channels: int = 3,  # H1, L1, V1
        base_filters: int = 32,
        output_dim: int = 256,
        input_length: int = 2048,  # Expected input sequence length
    ):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=32, stride=8, padding=12),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_filters, base_filters * 2, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(8),  # (N, 256, 8)
        )

        self.flatten_dim = base_filters * 8 * 8  # 256 * 8 = 2048

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Strain tensor of shape (batch, 3, seq_len)

        Returns:
            Features of shape (batch, output_dim)
        """
        assert x.ndim == 3, f"Expected 3D tensor (batch, channels, seq_len), got shape {x.shape}"
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"

        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


class PhysicsMLP(nn.Module):
    """
    MLP for processing auxiliary physics features.

    Takes 6D feature vector:
        [tau_HL, tau_HV, A_HL, A_HV, phi_HL, phi_HV]

    Adapted from Maxime's AuxiliaryMLP.
    """

    def __init__(
        self,
        in_features: int = 6,  # tau_HL, tau_HV, A_HL, A_HV, phi_HL, phi_HV
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()

        self.in_features = in_features
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Auxiliary features of shape (batch, in_features)

        Returns:
            Features of shape (batch, output_dim)
        """
        assert x.ndim == 2, f"Expected 2D tensor (batch, features), got shape {x.shape}"
        assert x.shape[1] == self.in_features, f"Expected {self.in_features} features, got {x.shape[1]}"

        return self.mlp(x)


class HybridBackbone(nn.Module):
    """
    Hybrid backbone combining strain CNN and physics MLP.

    This is the key innovation: combining end-to-end learning from raw data
    with explicit physics features.

    Architecture:
        - StrainCNN: 3 strains → 256D embedding
        - PhysicsMLP: 6 features → 32D embedding
        - Fusion: Concatenation → 288D combined embedding
    """

    def __init__(
        self,
        use_physics_features: bool = True,
        strain_feature_dim: int = 256,
        physics_feature_dim: int = 32,
        n_physics_features: int = 6,
        input_length: int = 2048,
    ):
        """
        Args:
            use_physics_features: If False, only use strain CNN (for ablation)
            strain_feature_dim: Dimensionality of strain features
            physics_feature_dim: Dimensionality of physics features
            n_physics_features: Number of input physics features (6)
            input_length: Expected strain sequence length
        """
        super().__init__()

        self.use_physics_features = use_physics_features

        self.strain_cnn = StrainCNN(
            in_channels=3,
            output_dim=strain_feature_dim,
            input_length=input_length,
        )

        if use_physics_features:
            self.physics_mlp = PhysicsMLP(
                in_features=n_physics_features,
                output_dim=physics_feature_dim,
            )
            self.output_dim = strain_feature_dim + physics_feature_dim
        else:
            self.physics_mlp = None
            self.output_dim = strain_feature_dim

    def forward(
        self,
        strain: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            strain: Strain tensor of shape (batch, 3, seq_len)
            physics_features: Physics features of shape (batch, 6) (optional)

        Returns:
            Combined features of shape (batch, output_dim)

        Note:
            If use_physics_features=True but physics_features is None,
            raises an error. If use_physics_features=False, physics_features
            is ignored even if provided.
        """
        strain_feat = self.strain_cnn(strain)

        if self.use_physics_features:
            if physics_features is None:
                raise ValueError("physics_features required when use_physics_features=True")

            physics_feat = self.physics_mlp(physics_features)
            combined = torch.cat([strain_feat, physics_feat], dim=1)
            return combined
        else:
            return strain_feat


if __name__ == "__main__":
    print("Testing backbone networks...")

    batch_size = 4
    seq_len = 2048

    print("\n1. Testing StrainCNN...")
    strain_cnn = StrainCNN(output_dim=256, input_length=seq_len)
    dummy_strain = torch.randn(batch_size, 3, seq_len)
    strain_features = strain_cnn(dummy_strain)
    print(f"   Input: {dummy_strain.shape}, Output: {strain_features.shape}")
    assert strain_features.shape == (batch_size, 256), "StrainCNN output shape mismatch!"

    print("\n2. Testing PhysicsMLP...")
    physics_mlp = PhysicsMLP(in_features=6, output_dim=32)
    dummy_physics = torch.randn(batch_size, 6)
    physics_features = physics_mlp(dummy_physics)
    print(f"   Input: {dummy_physics.shape}, Output: {physics_features.shape}")
    assert physics_features.shape == (batch_size, 32), "PhysicsMLP output shape mismatch!"

    print("\n3. Testing HybridBackbone (with physics)...")
    hybrid = HybridBackbone(use_physics_features=True, input_length=seq_len)
    combined_features = hybrid(dummy_strain, dummy_physics)
    print(f"   Output: {combined_features.shape}")
    assert combined_features.shape == (batch_size, 288), "HybridBackbone output shape mismatch!"

    print("\n4. Testing HybridBackbone (without physics)...")
    hybrid_no_physics = HybridBackbone(use_physics_features=False, input_length=seq_len)
    strain_only_features = hybrid_no_physics(dummy_strain)
    print(f"   Output: {strain_only_features.shape}")
    assert strain_only_features.shape == (batch_size, 256), "Strain-only output shape mismatch!"

    total_params = sum(p.numel() for p in hybrid.parameters())
    print(f"\n5. HybridBackbone total parameters: {total_params:,}")

    print("\nAll tests passed!")
