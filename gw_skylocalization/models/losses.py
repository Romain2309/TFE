"""
Loss functions for GW sky localization.

Provides:
- AngularLoss: Geodesic distance on sphere (proper metric for sky localization)
- CosineLoss: Smooth alternative to angular loss
- MultiTaskLoss: Combined loss for sky position + time delays
- Label smoothing for classification/probability maps (via standard CrossEntropyLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularLoss(nn.Module):
    """
    Angular distance loss for unit vectors on the sphere.

    Computes the geodesic distance:
        loss = arccos(pred · target)

    This is the proper metric for sky localization since it measures
    the actual angular separation on the sphere.

    Range: [0, π] radians = [0°, 180°]
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            eps: Small constant for numerical stability (clip dot product)
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute angular loss.

        Args:
            pred: Predicted unit vectors of shape (batch, 3)
            target: Target unit vectors of shape (batch, 3)

        Returns:
            Loss (scalar if reduction='mean' or 'sum', else (batch,))
        """
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        dot_product = torch.sum(pred * target, dim=1)

        dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)

        angular_dist = torch.acos(dot_product)

        if self.reduction == 'mean':
            return angular_dist.mean()
        elif self.reduction == 'sum':
            return angular_dist.sum()
        elif self.reduction == 'none':
            return angular_dist
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class CosineLoss(nn.Module):
    """
    Cosine similarity loss (alternative to angular loss).

    Computes:
        loss = 1 - cosine_similarity(pred, target)
             = 1 - (pred · target)

    Range: [0, 2]

    Advantages over AngularLoss:
    - Smoother gradients near the target
    - No arccos (faster computation, no numerical issues)

    Disadvantages:
    - Not a proper metric (doesn't measure angular distance)
    - Gradients vanish when very close to target
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine loss.

        Args:
            pred: Predicted unit vectors of shape (batch, 3)
            target: Target unit vectors of shape (batch, 3)

        Returns:
            Loss (scalar if reduction='mean' or 'sum', else (batch,))
        """
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        cosine_sim = torch.sum(pred * target, dim=1)

        loss = 1.0 - cosine_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for sky position + time delay prediction.

    Combines:
    - Main task: Sky position (angular loss)
    - Auxiliary task: Time delay prediction (MSE)

    Total loss = angular_loss + tau_weight * tau_loss

    The auxiliary task forces the network to learn physics-relevant features.
    """

    def __init__(
        self,
        tau_weight: float = 0.1,
        sky_loss_type: str = 'angular',
        reduction: str = 'mean'
    ):
        """
        Args:
            tau_weight: Weight for time delay loss (typically 0.1)
            sky_loss_type: 'angular' or 'cosine' for sky position loss
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.tau_weight = tau_weight
        self.reduction = reduction

        if sky_loss_type == 'angular':
            self.sky_loss = AngularLoss(reduction=reduction)
        elif sky_loss_type == 'cosine':
            self.sky_loss = CosineLoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid sky_loss_type: {sky_loss_type}")

        self.tau_loss_fn = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        pred_xyz: torch.Tensor,
        pred_tau: torch.Tensor,
        target_xyz: torch.Tensor,
        target_tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-task loss.

        Args:
            pred_xyz: Predicted unit vectors of shape (batch, 3)
            pred_tau: Predicted time delays of shape (batch, 2) - [tau_HL, tau_HV]
            target_xyz: Target unit vectors of shape (batch, 3)
            target_tau: Target time delays of shape (batch, 2)

        Returns:
            Combined loss (scalar if reduction='mean' or 'sum')
        """
        sky_loss = self.sky_loss(pred_xyz, target_xyz)

        tau_loss = self.tau_loss_fn(pred_tau, target_tau)

        total_loss = sky_loss + self.tau_weight * tau_loss

        return total_loss

    def compute_separate(
        self,
        pred_xyz: torch.Tensor,
        pred_tau: torch.Tensor,
        target_xyz: torch.Tensor,
        target_tau: torch.Tensor
    ) -> dict:
        """
        Compute losses separately (for logging).

        Returns:
            Dictionary with 'sky_loss', 'tau_loss', and 'total_loss'
        """
        sky_loss = self.sky_loss(pred_xyz, target_xyz)
        tau_loss = self.tau_loss_fn(pred_tau, target_tau)
        total_loss = sky_loss + self.tau_weight * tau_loss

        return {
            'sky_loss': sky_loss.item() if sky_loss.numel() == 1 else sky_loss,
            'tau_loss': tau_loss.item() if tau_loss.numel() == 1 else tau_loss,
            'total_loss': total_loss.item() if total_loss.numel() == 1 else total_loss,
        }


def angular_separation_degrees(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute angular separation in degrees (for metrics, not loss).

    Args:
        pred: Predicted unit vectors of shape (batch, 3) or (3,)
        target: Target unit vectors of shape (batch, 3) or (3,)

    Returns:
        Angular separations in degrees
    """
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    dot_product = torch.sum(pred * target, dim=-1)

    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angular_dist_rad = torch.acos(dot_product)

    angular_dist_deg = torch.rad2deg(angular_dist_rad)

    return angular_dist_deg


if __name__ == "__main__":
    print("Testing loss functions...")

    batch_size = 4

    pred_xyz = torch.randn(batch_size, 3)
    target_xyz = torch.randn(batch_size, 3)

    pred_xyz = F.normalize(pred_xyz, p=2, dim=1)
    target_xyz = F.normalize(target_xyz, p=2, dim=1)

    print("\n1. Testing AngularLoss...")
    angular_loss = AngularLoss(reduction='mean')
    loss_val = angular_loss(pred_xyz, target_xyz)
    print(f"   Loss: {loss_val.item():.6f} radians ({torch.rad2deg(loss_val).item():.2f}°)")
    assert 0 <= loss_val <= 3.15, "AngularLoss out of range [0, π]!"

    print("\n2. Testing CosineLoss...")
    cosine_loss = CosineLoss(reduction='mean')
    loss_val = cosine_loss(pred_xyz, target_xyz)
    print(f"   Loss: {loss_val.item():.6f}")
    assert 0 <= loss_val <= 2, "CosineLoss out of range [0, 2]!"

    print("\n3. Testing MultiTaskLoss...")
    pred_tau = torch.randn(batch_size, 2) * 0.01
    target_tau = torch.randn(batch_size, 2) * 0.01

    multitask_loss = MultiTaskLoss(tau_weight=0.1, sky_loss_type='angular')
    loss_val = multitask_loss(pred_xyz, pred_tau, target_xyz, target_tau)
    print(f"   Total loss: {loss_val.item():.6f}")

    losses = multitask_loss.compute_separate(pred_xyz, pred_tau, target_xyz, target_tau)
    print(f"   Sky loss: {losses['sky_loss']:.6f}")
    print(f"   Tau loss: {losses['tau_loss']:.6f}")

    print("\n4. Testing angular_separation_degrees...")
    v1 = torch.tensor([[1., 0., 0.]])  # (RA=0, Dec=0)
    v2 = torch.tensor([[0., 1., 0.]])  # (RA=90°, Dec=0)
    sep = angular_separation_degrees(v1, v2)
    print(f"   90° separation test: {sep.item():.2f}° (expected: 90.00°)")
    assert torch.allclose(sep, torch.tensor(90.0), atol=0.1), "Angular separation test failed!"

    v3 = torch.tensor([[1., 0., 0.]])
    v4 = torch.tensor([[1., 0., 0.]])
    sep = angular_separation_degrees(v3, v4)
    print(f"   0° separation test: {sep.item():.2f}° (expected: 0.00°)")
    assert torch.allclose(sep, torch.tensor(0.0), atol=0.1), "Perfect match test failed!"

    print("\nAll tests passed!")
