"""
Coordinate conversion utilities for gravitational wave sky localization.

Provides conversions between:
- (RA, Dec) spherical coordinates (radians)
- (x, y, z) unit vectors on the unit sphere
- HEALPix pixel indices

All angles are in radians unless otherwise specified.
"""

import numpy as np
import healpy as hp
from typing import Union, Tuple


def radec_to_xyz(ra: Union[float, np.ndarray], dec: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert (RA, Dec) to unit vector (x, y, z).

    Args:
        ra: Right ascension in radians [0, 2π)
        dec: Declination in radians [-π/2, π/2]

    Returns:
        xyz: Unit vector(s) of shape (..., 3)
            x = cos(dec) * cos(ra)
            y = cos(dec) * sin(ra)
            z = sin(dec)

    Examples:
        >>> xyz = radec_to_xyz(0.0, 0.0)  # (RA=0, Dec=0) → (1, 0, 0)
        >>> xyz = radec_to_xyz(np.pi/2, 0.0)  # (RA=π/2, Dec=0) → (0, 1, 0)
        >>> xyz = radec_to_xyz(0.0, np.pi/2)  # (RA=0, Dec=π/2) → (0, 0, 1)
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)

    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)

    if ra.ndim == 0:  # Scalar inputs
        return np.array([x, y, z])
    else:  # Array inputs
        return np.stack([x, y, z], axis=-1)


def xyz_to_radec(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert unit vector (x, y, z) to (RA, Dec).

    Args:
        xyz: Unit vector(s) of shape (..., 3)

    Returns:
        ra: Right ascension in radians [0, 2π)
        dec: Declination in radians [-π/2, π/2]

    Examples:
        >>> ra, dec = xyz_to_radec(np.array([1, 0, 0]))  # → (0, 0)
        >>> ra, dec = xyz_to_radec(np.array([0, 1, 0]))  # → (π/2, 0)
        >>> ra, dec = xyz_to_radec(np.array([0, 0, 1]))  # → (0, π/2)
    """
    xyz = np.asarray(xyz)

    if xyz.ndim == 1:  # Single vector
        x, y, z = xyz[0], xyz[1], xyz[2]
    else:  # Batch of vectors
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    dec = np.arcsin(np.clip(z, -1.0, 1.0))

    ra = np.arctan2(y, x)

    ra = np.where(ra < 0, ra + 2*np.pi, ra)

    return ra, dec


def radec_to_healpix(
    ra: Union[float, np.ndarray],
    dec: Union[float, np.ndarray],
    nside: int = 8,
    nest: bool = False
) -> Union[int, np.ndarray]:
    """
    Convert (RA, Dec) to HEALPix pixel index.

    Args:
        ra: Right ascension in radians [0, 2π)
        dec: Declination in radians [-π/2, π/2]
        nside: HEALPix resolution parameter (npix = 12*nside^2)
        nest: If True, use nested ordering; otherwise ring ordering

    Returns:
        pixel: HEALPix pixel index or indices

    Note:
        HEALPix uses (theta, phi) where theta = π/2 - dec, phi = ra
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)

    theta = np.pi / 2 - dec  # Co-latitude [0, π]
    phi = ra  # Azimuth [0, 2π)

    pixel = hp.ang2pix(nside, theta, phi, nest=nest)

    return pixel


def healpix_to_radec(
    pixel: Union[int, np.ndarray],
    nside: int = 8,
    nest: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert HEALPix pixel index to (RA, Dec).

    Args:
        pixel: HEALPix pixel index or indices
        nside: HEALPix resolution parameter
        nest: If True, use nested ordering; otherwise ring ordering

    Returns:
        ra: Right ascension in radians [0, 2π)
        dec: Declination in radians [-π/2, π/2]
    """
    pixel = np.asarray(pixel)

    theta, phi = hp.pix2ang(nside, pixel, nest=nest)

    dec = np.pi / 2 - theta  # Declination
    ra = phi  # Right ascension

    return ra, dec


def angular_separation(
    xyz1: np.ndarray,
    xyz2: np.ndarray,
    degrees: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute angular separation between unit vectors on the sphere.

    Uses the geodesic distance formula:
        angular_distance = arccos(xyz1 · xyz2)

    Args:
        xyz1: Unit vector(s) of shape (..., 3)
        xyz2: Unit vector(s) of shape (..., 3)
        degrees: If True, return result in degrees; otherwise radians

    Returns:
        Angular separation(s) in degrees or radians

    Examples:
        >>> sep = angular_separation([1, 0, 0], [0, 1, 0])  # 90 degrees
        >>> sep = angular_separation([1, 0, 0], [1, 0, 0])  # 0 degrees
        >>> sep = angular_separation([1, 0, 0], [-1, 0, 0])  # 180 degrees
    """
    xyz1 = np.asarray(xyz1)
    xyz2 = np.asarray(xyz2)

    dot_product = np.sum(xyz1 * xyz2, axis=-1)

    dot_product = np.clip(dot_product, -1.0, 1.0)

    angular_dist = np.arccos(dot_product)

    if degrees:
        return np.degrees(angular_dist)
    else:
        return angular_dist


def angular_separation_radec(
    ra1: Union[float, np.ndarray],
    dec1: Union[float, np.ndarray],
    ra2: Union[float, np.ndarray],
    dec2: Union[float, np.ndarray],
    degrees: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute angular separation between (RA, Dec) positions.

    Uses the Haversine formula for numerical stability:
        d = 2 * arcsin(sqrt(sin²(Δdec/2) + cos(dec1)*cos(dec2)*sin²(Δra/2)))

    Args:
        ra1, dec1: First position in radians
        ra2, dec2: Second position in radians
        degrees: If True, return result in degrees

    Returns:
        Angular separation(s) in degrees or radians
    """
    ra1, dec1 = np.asarray(ra1), np.asarray(dec1)
    ra2, dec2 = np.asarray(ra2), np.asarray(dec2)

    delta_ra = ra2 - ra1
    delta_dec = dec2 - dec1

    a = np.sin(delta_dec/2)**2 + \
        np.cos(dec1) * np.cos(dec2) * np.sin(delta_ra/2)**2

    angular_dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    if degrees:
        return np.degrees(angular_dist)
    else:
        return angular_dist


def radec_deg_to_xyz(ra_deg: Union[float, np.ndarray], dec_deg: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convenience function: Convert (RA, Dec) in degrees to unit vector.

    Args:
        ra_deg: Right ascension in degrees [0, 360)
        dec_deg: Declination in degrees [-90, 90]

    Returns:
        xyz: Unit vector(s) of shape (..., 3)
    """
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return radec_to_xyz(ra_rad, dec_rad)


def xyz_to_radec_deg(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: Convert unit vector to (RA, Dec) in degrees.

    Args:
        xyz: Unit vector(s) of shape (..., 3)

    Returns:
        ra_deg: Right ascension in degrees [0, 360)
        dec_deg: Declination in degrees [-90, 90]
    """
    ra, dec = xyz_to_radec(xyz)
    return np.degrees(ra), np.degrees(dec)


def get_healpix_npix(nside: int) -> int:
    """
    Get the number of pixels for a given HEALPix nside.

    Args:
        nside: HEALPix resolution parameter (must be power of 2)

    Returns:
        Number of pixels = 12 * nside^2

    Examples:
        >>> get_healpix_npix(1)   # 12
        >>> get_healpix_npix(2)   # 48
        >>> get_healpix_npix(4)   # 192
        >>> get_healpix_npix(8)   # 768
        >>> get_healpix_npix(16)  # 3072
    """
    return hp.nside2npix(nside)


def normalize_xyz(xyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize vectors to unit length.

    Useful for ensuring predictions from neural networks lie on the unit sphere.

    Args:
        xyz: Vectors of shape (..., 3)
        eps: Small constant to avoid division by zero

    Returns:
        Normalized unit vectors of shape (..., 3)
    """
    xyz = np.asarray(xyz)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    return xyz / (norm + eps)


if __name__ == "__main__":
    print("Testing coordinate conversions...")

    ra_test, dec_test = np.pi/4, np.pi/6
    xyz = radec_to_xyz(ra_test, dec_test)
    ra_back, dec_back = xyz_to_radec(xyz)
    print(f"Round-trip test: RA {ra_test:.6f} → {ra_back:.6f}, Dec {dec_test:.6f} → {dec_back:.6f}")
    assert np.allclose(ra_test, ra_back) and np.allclose(dec_test, dec_back), "Round-trip failed!"

    xyz1 = np.array([1, 0, 0])
    xyz2 = np.array([0, 1, 0])
    sep = angular_separation(xyz1, xyz2, degrees=True)
    print(f"Angular separation (90° expected): {sep:.2f}°")
    assert np.allclose(sep, 90.0), "Angular separation test failed!"

    nside = 8
    npix = get_healpix_npix(nside)
    print(f"HEALPix nside={nside} → {npix} pixels")

    ra_hp, dec_hp = 0.0, 0.0
    pix = radec_to_healpix(ra_hp, dec_hp, nside=nside)
    ra_hp_back, dec_hp_back = healpix_to_radec(pix, nside=nside)
    print(f"HEALPix round-trip: pixel {pix}, RA {ra_hp:.6f} → {ra_hp_back:.6f}, Dec {dec_hp:.6f} → {dec_hp_back:.6f}")

    print("\nAll tests passed!")
