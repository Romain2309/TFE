"""
Complete neural network architectures for GW sky localization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import healpy as hp

from .backbones import HybridBackbone
from .heads import (
    RegressionHead,
    ClassificationHead,
    ProbabilityHead,
    MultiTaskHead,
)


class SkyClassifier(nn.Module):
    """
    Sky localization as HEALPix pixel classification.
    """

    def __init__(
        self,
        n_pixels: int = 768,  # HEALPix nside=8 → 768 pixels
        use_physics_features: bool = True,
        strain_feature_dim: int = 256,
        physics_feature_dim: int = 32,
        n_physics_features: int = 6,
        input_length: int = 2048,
    ):
        """
        Args:
            n_pixels: Number of HEALPix pixels (12 * nside^2)
            use_physics_features: If True, use hybrid backbone; else strain-only
            strain_feature_dim: Dimensionality of strain features
            physics_feature_dim: Dimensionality of physics features
            n_physics_features: Number of input physics features
            input_length: Expected strain sequence length
        """
        super().__init__()

        self.n_pixels = n_pixels
        self.use_physics_features = use_physics_features

        self.backbone = HybridBackbone(
            use_physics_features=use_physics_features,
            strain_feature_dim=strain_feature_dim,
            physics_feature_dim=physics_feature_dim,
            n_physics_features=n_physics_features,
            input_length=input_length,
        )

        self.head = ClassificationHead(
            in_features=self.backbone.output_dim,
            n_pixels=n_pixels,
        )

    def forward(
        self,
        strain: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            strain: Strain tensor of shape (batch, 3, seq_len)
            physics_features: Physics features of shape (batch, 6) (optional)

        Returns:
            Logits of shape (batch, n_pixels)
        """
        features = self.backbone(strain, physics_features)

        logits = self.head(features)

        return logits


class SkyRegressor(nn.Module):
    """
    Sky localization as unit vector regression.
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
            use_physics_features: If True, use hybrid backbone; else strain-only
            strain_feature_dim: Dimensionality of strain features
            physics_feature_dim: Dimensionality of physics features
            n_physics_features: Number of input physics features (6)
            input_length: Expected strain sequence length
        """
        super().__init__()

        self.use_physics_features = use_physics_features

        self.backbone = HybridBackbone(
            use_physics_features=use_physics_features,
            strain_feature_dim=strain_feature_dim,
            physics_feature_dim=physics_feature_dim,
            n_physics_features=n_physics_features,
            input_length=input_length,
        )

        self.head = RegressionHead(
            in_features=self.backbone.output_dim,
        )

    def forward(
        self,
        strain: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            strain: Strain tensor of shape (batch, 3, seq_len)
            physics_features: Physics features of shape (batch, 6) (optional)

        Returns:
            Unit vectors of shape (batch, 3) with ||xyz|| = 1
        """
        features = self.backbone(strain, physics_features)

        xyz = self.head(features)

        return xyz


class SkyMultiTask(nn.Module):
    """
    Multi-task learning: sky position + time delay prediction.
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
            use_physics_features: If True, use hybrid backbone; else strain-only
            strain_feature_dim: Dimensionality of strain features
            physics_feature_dim: Dimensionality of physics features
            n_physics_features: Number of input physics features (6)
            input_length: Expected strain sequence length
        """
        super().__init__()

        self.use_physics_features = use_physics_features

        self.backbone = HybridBackbone(
            use_physics_features=use_physics_features,
            strain_feature_dim=strain_feature_dim,
            physics_feature_dim=physics_feature_dim,
            n_physics_features=n_physics_features,
            input_length=input_length,
        )

        self.head = MultiTaskHead(
            in_features=self.backbone.output_dim,
        )

    def forward(
        self,
        strain: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            strain: Strain tensor of shape (batch, 3, seq_len)
            physics_features: Physics features of shape (batch, 6) (optional)

        Returns:
            xyz: Unit vectors of shape (batch, 3)
            tau: Time delays of shape (batch, 2) - [tau_HL, tau_HV]
        """
        features = self.backbone(strain, physics_features)

        xyz, tau = self.head(features)

        return xyz, tau


class SkyProbabilityMap(nn.Module):
    """
    Sky localization as probability distribution over HEALPix pixels.
    """

    def __init__(
        self,
        n_pixels: int = 768,  # HEALPix nside=8 → 768 pixels
        use_physics_features: bool = True,
        strain_feature_dim: int = 256,
        physics_feature_dim: int = 32,
        n_physics_features: int = 6,
        input_length: int = 2048,
    ):
        """
        Args:
            n_pixels: Number of HEALPix pixels (12 * nside^2)
            use_physics_features: If True, use hybrid backbone; else strain-only
            strain_feature_dim: Dimensionality of strain features
            physics_feature_dim: Dimensionality of physics features
            n_physics_features: Number of input physics features (6)
            input_length: Expected strain sequence length
        """
        super().__init__()

        self.n_pixels = n_pixels
        self.use_physics_features = use_physics_features

        self.backbone = HybridBackbone(
            use_physics_features=use_physics_features,
            strain_feature_dim=strain_feature_dim,
            physics_feature_dim=physics_feature_dim,
            n_physics_features=n_physics_features,
            input_length=input_length,
        )

        self.head = ProbabilityHead(
            in_features=self.backbone.output_dim,
            n_pixels=n_pixels,
        )

    def forward(
        self,
        strain: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            strain: Strain tensor of shape (batch, 3, seq_len)
            physics_features: Physics features of shape (batch, 6) (optional)
            return_probs: If True, return probabilities; if False, return logits
                         (default False for training with CrossEntropyLoss)

        Returns:
            If return_probs=True: Probability distribution of shape (batch, n_pixels)
            If return_probs=False: Logits of shape (batch, n_pixels)
        """
        features = self.backbone(strain, physics_features)

        output = self.head(features, return_probs=return_probs)

        return output


"""
if __name__ == "__main__":
    print("Testing complete neural network models...")

    batch_size = 2
    seq_len = 2048
    n_pixels = 768  # nside=8

    dummy_strain = torch.randn(batch_size, 3, seq_len)
    dummy_physics = torch.randn(batch_size, 6)

    print("\n1. Testing SkyClassifier...")
    classifier = SkyClassifier(n_pixels=n_pixels, use_physics_features=True, input_length=seq_len)
    logits = classifier(dummy_strain, dummy_physics)
    print(f"   Output shape: {logits.shape} (expected: ({batch_size}, {n_pixels}))")
    assert logits.shape == (batch_size, n_pixels)

    classifier_no_physics = SkyClassifier(n_pixels=n_pixels, use_physics_features=False, input_length=seq_len)
    logits_no_physics = classifier_no_physics(dummy_strain)
    print(f"   Output (no physics): {logits_no_physics.shape}")
    assert logits_no_physics.shape == (batch_size, n_pixels)

    print("\n2. Testing SkyRegressor...")
    regressor = SkyRegressor(use_physics_features=True, input_length=seq_len)
    xyz = regressor(dummy_strain, dummy_physics)
    print(f"   Output shape: {xyz.shape} (expected: ({batch_size}, 3))")
    assert xyz.shape == (batch_size, 3)

    norms = torch.norm(xyz, p=2, dim=1)
    print(f"   Norms: {norms.tolist()} (should all be ~1.0)")
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

    print("\n3. Testing SkyMultiTask...")
    multitask = SkyMultiTask(use_physics_features=True, input_length=seq_len)
    xyz_mt, tau = multitask(dummy_strain, dummy_physics)
    print(f"   Sky output shape: {xyz_mt.shape} (expected: ({batch_size}, 3))")
    print(f"   Tau output shape: {tau.shape} (expected: ({batch_size}, 2))")
    assert xyz_mt.shape == (batch_size, 3)
    assert tau.shape == (batch_size, 2)

    norms_mt = torch.norm(xyz_mt, p=2, dim=1)
    assert torch.allclose(norms_mt, torch.ones(batch_size), atol=1e-5)

    print("\n4. Testing SkyProbabilityMap...")
    probmap = SkyProbabilityMap(n_pixels=n_pixels, use_physics_features=True, input_length=seq_len)
    probs = probmap(dummy_strain, dummy_physics)
    print(f"   Output shape: {probs.shape} (expected: ({batch_size}, {n_pixels}))")
    assert probs.shape == (batch_size, n_pixels)

    prob_sums = probs.sum(dim=1)
    print(f"   Probability sums: {prob_sums.tolist()} (should all be ~1.0)")
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)

    print("\n5. Parameter counts:")
    print(f"   SkyClassifier: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"   SkyRegressor: {sum(p.numel() for p in regressor.parameters()):,}")
    print(f"   SkyMultiTask: {sum(p.numel() for p in multitask.parameters()):,}")
    print(f"   SkyProbabilityMap: {sum(p.numel() for p in probmap.parameters()):,}")

    print("\nAll tests passed!")
"""
