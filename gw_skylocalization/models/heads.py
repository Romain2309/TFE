"""
Task-specific prediction heads for GW sky localization.

Provides:
- RegressionHead: Unit vector (x, y, z) prediction
- ClassificationHead: HEALPix pixel classification
- ProbabilityHead: HEALPix probability distribution
- MultiTaskHead: Combined sky position + time delay prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RegressionHead(nn.Module):
    """
    Regression head for predicting unit vector (x, y, z) on sphere.

    Outputs are normalized to unit length to ensure valid sky positions.
    """

    def __init__(
        self,
        in_features: int = 288,  # From HybridBackbone
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # (x, y, z)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (batch, in_features)

        Returns:
            Unit vectors of shape (batch, 3) with ||xyz|| = 1
        """
        xyz = self.regressor(x)

        xyz = F.normalize(xyz, p=2, dim=1)

        return xyz


class ClassificationHead(nn.Module):
    """
    Classification head for HEALPix pixel prediction.

    Treats sky localization as discrete classification into HEALPix pixels.
    """

    def __init__(
        self,
        in_features: int = 288,
        n_pixels: int = 768,  # HEALPix nside=8 → 768 pixels
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.n_pixels = n_pixels

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_pixels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (batch, in_features)

        Returns:
            Logits of shape (batch, n_pixels)
        """
        logits = self.classifier(x)
        return logits


class ProbabilityHead(nn.Module):
    """
    Probability head for HEALPix probability distribution.

    Outputs a full probability distribution over the sky, enabling:
    - Uncertainty quantification
    - Credible region estimation
    - Bimodality detection (mirror degeneracy)
    """

    def __init__(
        self,
        in_features: int = 288,
        n_pixels: int = 768,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.n_pixels = n_pixels

        self.prob_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_pixels),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (batch, in_features)
            return_probs: If True, return probabilities; if False, return logits

        Returns:
            If return_probs=True: Probability distribution of shape (batch, n_pixels)
                                  where sum over pixels = 1
            If return_probs=False: Logits of shape (batch, n_pixels)
        """
        logits = self.prob_net(x)

        if return_probs:
            return F.softmax(logits, dim=1)
        else:
            return logits


class MultiTaskHead(nn.Module):
    """
    Multi-task head for sky position + time delay prediction.

    Auxiliary task: Predict time delays (tau_HL, tau_HV) to enforce learning
    of physics-relevant features.

    Main task: Predict unit vector sky position.
    """

    def __init__(
        self,
        in_features: int = 288,
        shared_hidden_dim: int = 128,
        sky_hidden_dim: int = 64,
        tau_hidden_dim: int = 32,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_features, shared_hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.sky_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, sky_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sky_hidden_dim, 3),  # (x, y, z)
        )

        self.tau_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, tau_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(tau_hidden_dim, 2),  # (tau_HL, tau_HV)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Features of shape (batch, in_features)

        Returns:
            xyz: Unit vectors of shape (batch, 3)
            tau: Time delays of shape (batch, 2) - [tau_HL, tau_HV]
        """
        shared_feat = self.shared(x)

        xyz = self.sky_head(shared_feat)
        xyz = F.normalize(xyz, p=2, dim=1)  # Normalize to unit length

        tau = self.tau_head(shared_feat)

        return xyz, tau


if __name__ == "__main__":
    print("Testing prediction heads...")

    batch_size = 4
    in_features = 288

    dummy_features = torch.randn(batch_size, in_features)

    print("\n1. Testing RegressionHead...")
    reg_head = RegressionHead(in_features=in_features)
    xyz = reg_head(dummy_features)
    print(f"   Input: {dummy_features.shape}, Output: {xyz.shape}")
    assert xyz.shape == (batch_size, 3), "RegressionHead output shape mismatch!"

    norms = torch.norm(xyz, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), "Vectors not normalized!"
    print(f"   Norms: {norms} (all should be 1.0)")

    print("\n2. Testing ClassificationHead...")
    n_pixels = 768
    cls_head = ClassificationHead(in_features=in_features, n_pixels=n_pixels)
    logits = cls_head(dummy_features)
    print(f"   Input: {dummy_features.shape}, Output: {logits.shape}")
    assert logits.shape == (batch_size, n_pixels), "ClassificationHead output shape mismatch!"

    print("\n3. Testing ProbabilityHead...")
    prob_head = ProbabilityHead(in_features=in_features, n_pixels=n_pixels)
    probs = prob_head(dummy_features)
    print(f"   Input: {dummy_features.shape}, Output: {probs.shape}")
    assert probs.shape == (batch_size, n_pixels), "ProbabilityHead output shape mismatch!"

    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1!"
    print(f"   Probability sums: {prob_sums} (all should be 1.0)")

    print("\n4. Testing MultiTaskHead...")
    multitask_head = MultiTaskHead(in_features=in_features)
    xyz_mt, tau = multitask_head(dummy_features)
    print(f"   Input: {dummy_features.shape}")
    print(f"   Sky output: {xyz_mt.shape}")
    print(f"   Time delay output: {tau.shape}")
    assert xyz_mt.shape == (batch_size, 3), "MultiTaskHead sky output shape mismatch!"
    assert tau.shape == (batch_size, 2), "MultiTaskHead tau output shape mismatch!"

    norms_mt = torch.norm(xyz_mt, p=2, dim=1)
    assert torch.allclose(norms_mt, torch.ones(batch_size), atol=1e-5), "MultiTask vectors not normalized!"
    print(f"   Sky position norms: {norms_mt} (all should be 1.0)")

    print("\n5. Parameter counts:")
    print(f"   RegressionHead: {sum(p.numel() for p in reg_head.parameters()):,}")
    print(f"   ClassificationHead: {sum(p.numel() for p in cls_head.parameters()):,}")
    print(f"   ProbabilityHead: {sum(p.numel() for p in prob_head.parameters()):,}")
    print(f"   MultiTaskHead: {sum(p.numel() for p in multitask_head.parameters()):,}")

    print("\nAll tests passed!")
