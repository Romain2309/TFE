"""
Evaluation metrics for gravitational wave sky localization.

Provides:
- Angular error statistics (for regression models)
- HEALPix metrics (for classification/probability models)
- Searched area and credible regions (standard GW astronomy metrics)
- Bimodality detection (mirror degeneracy)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


def compute_angular_errors(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
    degrees: bool = True
) -> np.ndarray:
    """
    Compute angular errors between predicted and target unit vectors.

    Args:
        pred_xyz: (n, 3) predicted unit vectors
        target_xyz: (n, 3) target unit vectors
        degrees: Return errors in degrees (else radians)

    Returns:
        (n,) array of angular errors
    """
    pred_xyz = pred_xyz / np.linalg.norm(pred_xyz, axis=1, keepdims=True)
    target_xyz = target_xyz / np.linalg.norm(target_xyz, axis=1, keepdims=True)

    dot_product = np.sum(pred_xyz * target_xyz, axis=1)

    dot_product = np.clip(dot_product, -1.0, 1.0)

    angular_dist = np.arccos(dot_product)

    if degrees:
        angular_dist = np.degrees(angular_dist)

    return angular_dist


def angular_error_statistics(errors: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for angular errors.

    Args:
        errors: (n,) array of angular errors in degrees

    Returns:
        Dictionary with statistics:
        - median, mean, std
        - min, max
        - p10, p25, p75, p90, p95, p99 (percentiles)
    """
    return {
        'median': float(np.median(errors)),
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'p10': float(np.percentile(errors, 10)),
        'p25': float(np.percentile(errors, 25)),
        'p75': float(np.percentile(errors, 75)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'p99': float(np.percentile(errors, 99)),
    }


def searched_area(
    probs: np.ndarray,
    true_pixel: int,
    nside: int,
    credible_levels: Tuple[float, ...] = (0.5, 0.9),
) -> Dict[str, float]:
    """
    Compute searched area metrics for a probability sky map.

    Standard metric in GW astronomy: how much area must be searched
    to find the true source location.

    Args:
        probs: (n_pixels,) probability per pixel
        true_pixel: Index of true sky position
        nside: HEALPix nside parameter
        credible_levels: Credible levels to compute (e.g., 0.5, 0.9)

    Returns:
        Dictionary with:
        - searched_area_deg2: Area needed to reach true position
        - prob_at_true: Probability at true pixel
        - cumprob_at_true: Cumulative probability up to true pixel
        - true_rank: Rank of true pixel in sorted probabilities
        - area_50_deg2, area_90_deg2: Credible region areas
        - true_in_50, true_in_90: Whether true position in credible region
    """
    if not HAS_HEALPY:
        raise ImportError("healpy required for HEALPix metrics")

    n_pixels = len(probs)

    if true_pixel < 0 or true_pixel >= n_pixels:
        raise ValueError(f"true_pixel {true_pixel} out of range [0, {n_pixels})")
    if np.any(probs < 0):
        raise ValueError("Negative probabilities found")

    probs_sum = probs.sum()
    if probs_sum <= 0:
        raise ValueError("Probabilities sum to zero or negative")
    probs = probs / probs_sum

    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])

    true_rank = np.where(sorted_idx == true_pixel)[0]
    if len(true_rank) == 0:
        raise ValueError(f"true_pixel {true_pixel} not found")
    true_rank = int(true_rank[0])

    pixel_area_deg2 = hp.nside2pixarea(nside, degrees=True)

    searched_area_deg2 = (true_rank + 1) * pixel_area_deg2

    result = {
        'searched_area_deg2': float(searched_area_deg2),
        'prob_at_true': float(probs[true_pixel]),
        'cumprob_at_true': float(cumsum[true_rank]),
        'true_rank': int(true_rank),
    }

    for level in credible_levels:
        n_pixels_level = int(np.searchsorted(cumsum, level)) + 1
        n_pixels_level = min(n_pixels_level, n_pixels)
        area_deg2 = n_pixels_level * pixel_area_deg2

        level_pct = int(level * 100)
        result[f'area_{level_pct}_deg2'] = float(area_deg2)
        result[f'true_in_{level_pct}'] = bool(true_rank < n_pixels_level)

    return result


def searched_area_batch(
    probs_batch: np.ndarray,
    true_pixels: np.ndarray,
    nside: int,
    credible_levels: Tuple[float, ...] = (0.5, 0.9),
) -> Dict[str, np.ndarray]:
    """
    Compute searched area metrics for a batch of probability maps.

    Args:
        probs_batch: (batch, n_pixels) probabilities
        true_pixels: (batch,) true pixel indices
        nside: HEALPix nside parameter
        credible_levels: Credible levels to compute

    Returns:
        Dictionary with arrays of metrics (one per sample)
    """
    batch_size = probs_batch.shape[0]

    keys = ['searched_area_deg2', 'prob_at_true', 'cumprob_at_true', 'true_rank']
    for level in credible_levels:
        level_pct = int(level * 100)
        keys.extend([f'area_{level_pct}_deg2', f'true_in_{level_pct}'])

    results = {key: [] for key in keys}

    for i in range(batch_size):
        sample_result = searched_area(
            probs_batch[i],
            true_pixels[i],
            nside,
            credible_levels
        )
        for key in keys:
            results[key].append(sample_result[key])

    for key in keys:
        results[key] = np.array(results[key])

    return results


def find_skymap_modes(
    probs: np.ndarray,
    nside: int,
    min_separation_deg: float = 90.0,
    threshold: float = 0.01,
) -> List[Dict]:
    """
    Find modes (peaks) in a probability sky map.

    Used to detect bimodality from mirror degeneracy.

    Args:
        probs: (n_pixels,) probability per pixel
        nside: HEALPix nside parameter
        min_separation_deg: Minimum separation between modes (degrees)
        threshold: Minimum probability to consider as mode

    Returns:
        List of modes, each a dict with:
        - pixel: pixel index
        - prob: probability
        - ra_deg, dec_deg: position in degrees
    """
    if not HAS_HEALPY:
        raise ImportError("healpy required")

    probs = probs / probs.sum()

    sorted_idx = np.argsort(probs)[::-1]

    modes = []

    for pix in sorted_idx:
        prob = probs[pix]

        if prob < threshold:
            break

        too_close = False
        for existing_mode in modes:
            theta1, phi1 = hp.pix2ang(nside, pix)
            theta2, phi2 = hp.pix2ang(nside, existing_mode['pixel'])

            sep = hp.rotator.angdist([theta1, phi1], [theta2, phi2], lonlat=False)
            sep_deg = np.degrees(sep)

            if sep_deg < min_separation_deg:
                too_close = True
                break

        if not too_close:
            theta, phi = hp.pix2ang(nside, pix)
            dec_rad = np.pi / 2 - theta
            ra_rad = phi

            modes.append({
                'pixel': int(pix),
                'prob': float(prob),
                'ra_deg': float(np.degrees(ra_rad)),
                'dec_deg': float(np.degrees(dec_rad)),
            })

    return modes


def is_bimodal(
    probs: np.ndarray,
    nside: int,
    secondary_threshold: float = 0.3,
    min_separation_deg: float = 90.0,
) -> bool:
    """
    Detect if a sky map shows bimodality (e.g., mirror degeneracy).

    Args:
        probs: (n_pixels,) probability per pixel
        nside: HEALPix nside parameter
        secondary_threshold: Min ratio of secondary mode to primary
        min_separation_deg: Min separation for modes to count

    Returns:
        True if bimodal, False otherwise
    """
    modes = find_skymap_modes(probs, nside, min_separation_deg=min_separation_deg)

    if len(modes) < 2:
        return False

    primary_prob = modes[0]['prob']
    secondary_prob = modes[1]['prob']

    return (secondary_prob / primary_prob) >= secondary_threshold


def compute_time_delay_metrics(
    pred_tau: np.ndarray,
    target_tau: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for time delay prediction (MultiTask model).

    Args:
        pred_tau: (n, 2) predicted [tau_HL, tau_HV] in seconds
        target_tau: (n, 2) target time delays in seconds

    Returns:
        Dictionary with:
        - mae: Mean absolute error
        - mae_ms: MAE in milliseconds
        - rmse: Root mean squared error
        - rmse_ms: RMSE in milliseconds
        - mae_HL, mae_HV: Per-detector MAE
    """
    mae = np.mean(np.abs(pred_tau - target_tau))
    rmse = np.sqrt(np.mean((pred_tau - target_tau) ** 2))

    mae_HL = np.mean(np.abs(pred_tau[:, 0] - target_tau[:, 0]))
    mae_HV = np.mean(np.abs(pred_tau[:, 1] - target_tau[:, 1]))

    return {
        'mae': float(mae),
        'mae_ms': float(mae * 1000),
        'rmse': float(rmse),
        'rmse_ms': float(rmse * 1000),
        'mae_HL': float(mae_HL),
        'mae_HV': float(mae_HV),
        'mae_HL_ms': float(mae_HL * 1000),
        'mae_HV_ms': float(mae_HV * 1000),
    }


def classification_accuracy(
    pred_pixels: np.ndarray,
    true_pixels: np.ndarray,
) -> float:
    """
    Compute pixel classification accuracy.

    Args:
        pred_pixels: (n,) predicted pixel indices
        true_pixels: (n,) true pixel indices

    Returns:
        Accuracy (fraction correct)
    """
    return float(np.mean(pred_pixels == true_pixels))


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    print("\n1. Testing angular error metrics...")
    pred_xyz = np.random.randn(100, 3)
    pred_xyz = pred_xyz / np.linalg.norm(pred_xyz, axis=1, keepdims=True)
    target_xyz = np.random.randn(100, 3)
    target_xyz = target_xyz / np.linalg.norm(target_xyz, axis=1, keepdims=True)

    errors = compute_angular_errors(pred_xyz, target_xyz, degrees=True)
    stats = angular_error_statistics(errors)

    print(f"   Median error: {stats['median']:.2f}°")
    print(f"   Mean error: {stats['mean']:.2f}°")
    print(f"   90th percentile: {stats['p90']:.2f}°")

    if HAS_HEALPY:
        print("\n2. Testing HEALPix metrics...")
        nside = 8
        n_pixels = 12 * nside ** 2

        probs = np.random.rand(n_pixels) * 0.01
        probs[100] = 0.5  # Strong peak
        probs = probs / probs.sum()

        sa_metrics = searched_area(probs, true_pixel=100, nside=nside)
        print(f"   Searched area: {sa_metrics['searched_area_deg2']:.2f} deg²")
        print(f"   Probability at true: {sa_metrics['prob_at_true']:.4f}")
        print(f"   True in 90% region: {sa_metrics['true_in_90']}")

        print("\n3. Testing bimodality detection...")
        probs_bimodal = probs.copy()
        probs_bimodal[500] = 0.3  # Secondary peak
        probs_bimodal = probs_bimodal / probs_bimodal.sum()

        is_bi = is_bimodal(probs_bimodal, nside=nside)
        print(f"   Is bimodal: {is_bi}")

        modes = find_skymap_modes(probs_bimodal, nside=nside)
        print(f"   Found {len(modes)} modes:")
        for i, mode in enumerate(modes):
            print(f"     Mode {i+1}: pixel={mode['pixel']}, prob={mode['prob']:.4f}")

    print("\n4. Testing time delay metrics...")
    pred_tau = np.random.randn(100, 2) * 0.01
    target_tau = np.random.randn(100, 2) * 0.01

    tau_metrics = compute_time_delay_metrics(pred_tau, target_tau)
    print(f"   MAE: {tau_metrics['mae_ms']:.2f} ms")
    print(f"   RMSE: {tau_metrics['rmse_ms']:.2f} ms")

    print("\nAll tests passed!")
