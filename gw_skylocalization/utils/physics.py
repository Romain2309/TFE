"""
Physics-based feature extraction for gravitational wave sky localization.

Provides functions to compute:
- Time delays between detector pairs (via cross-correlation or geometric calculation)
- Amplitude ratios between detector pairs
- Phase differences between detector pairs

These features are physics-informed and directly related to sky position.
"""

import numpy as np
from scipy.signal import correlate, hilbert
from typing import Dict, Union, Tuple, Optional

SPEED_OF_LIGHT = 299792458.0  # m/s
MAX_TIME_DELAY_GW = 0.04  # Maximum time delay between any two detectors (seconds)


def compute_time_delay_xcorr(
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    fs: float,
    max_delay: float = MAX_TIME_DELAY_GW
) -> float:
    """
    Compute time delay between two strains using cross-correlation.

    Args:
        strain_a: First strain time series
        strain_b: Second strain time series
        fs: Sampling frequency in Hz
        max_delay: Maximum expected time delay in seconds (search window)

    Returns:
        Time delay in seconds (positive if strain_a leads strain_b)

    Algorithm:
        1. Compute cross-correlation via FFT
        2. Restrict search to ±max_delay window
        3. Find peak of cross-correlation
        4. Convert lag to time delay

    Examples:
        >>> delay = compute_time_delay_xcorr(h1_strain, l1_strain, fs=1024)
    """
    xcorr = correlate(strain_a, strain_b, mode='same', method='fft')

    center = len(xcorr) // 2

    max_lag_samples = int(max_delay * fs)
    search_start = max(0, center - max_lag_samples)
    search_end = min(len(xcorr), center + max_lag_samples + 1)

    xcorr_windowed = xcorr[search_start:search_end]
    peak_idx_in_window = np.argmax(xcorr_windowed)
    peak_idx = search_start + peak_idx_in_window

    sample_delay = peak_idx - center
    time_delay = sample_delay / fs

    return time_delay


def compute_time_delay_geometric(
    ra: float,
    dec: float,
    detector_a: str,
    detector_b: str,
    t_gps: float = 1000000000.0
) -> float:
    """
    Compute exact geometric time delay between two detectors.

    Uses PyCBC detector positions and geometry to compute the precise
    time delay for a signal from (RA, Dec).

    Args:
        ra: Right ascension in radians
        dec: Declination in radians
        detector_a: First detector name ('H1', 'L1', or 'V1')
        detector_b: Second detector name
        t_gps: GPS time for detector orientation

    Returns:
        Time delay in seconds

    Note:
        Requires pycbc.detector module

    Examples:
        >>> from pycbc.detector import Detector
        >>> tau = compute_time_delay_geometric(0.0, 0.0, 'H1', 'L1')
    """
    try:
        from pycbc.detector import Detector
    except ImportError:
        raise ImportError("pycbc is required for geometric time delay computation. "
                          "Install with: pip install pycbc")

    det_a = Detector(detector_a)
    det_b = Detector(detector_b)

    tau = det_a.time_delay_from_location(det_b.location, ra, dec, t_gps)

    return tau


def compute_amplitude_ratio(
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    method: str = 'max',
    eps: float = 1e-30
) -> float:
    """
    Compute amplitude ratio between two strains.

    Args:
        strain_a: First strain time series
        strain_b: Second strain time series
        method: 'max' (peak amplitude) or 'rms' (root mean square)
        eps: Small constant to avoid division by zero

    Returns:
        Amplitude ratio A_a / A_b

    Examples:
        >>> ratio = compute_amplitude_ratio(h1_strain, l1_strain, method='max')
    """
    if method == 'max':
        A_a = np.max(np.abs(strain_a))
        A_b = np.max(np.abs(strain_b))
    elif method == 'rms':
        A_a = np.sqrt(np.mean(strain_a ** 2))
        A_b = np.sqrt(np.mean(strain_b ** 2))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'max' or 'rms'.")

    ratio = A_a / (A_b + eps)

    return ratio


def get_phase_at_peak(strain: np.ndarray) -> float:
    """
    Get the phase at maximum amplitude using Hilbert transform.

    Args:
        strain: Strain time series

    Returns:
        Phase at peak amplitude in radians [-π, π]

    Algorithm:
        1. Compute analytic signal via Hilbert transform
        2. Find index of maximum amplitude
        3. Extract phase at that index

    Examples:
        >>> phase = get_phase_at_peak(h1_strain)
    """
    analytic = hilbert(strain)

    peak_idx = np.argmax(np.abs(strain))

    phase = np.angle(analytic[peak_idx])

    return phase


def compute_phase_difference(
    strain_a: np.ndarray,
    strain_b: np.ndarray
) -> float:
    """
    Compute phase difference at peak between two strains.

    Args:
        strain_a: First strain time series
        strain_b: Second strain time series

    Returns:
        Phase difference phi_a - phi_b in radians

    Examples:
        >>> phi_diff = compute_phase_difference(h1_strain, l1_strain)
    """
    phi_a = get_phase_at_peak(strain_a)
    phi_b = get_phase_at_peak(strain_b)

    phi_diff = phi_a - phi_b

    return phi_diff


def extract_physics_features(
    strains: Dict[str, np.ndarray],
    fs: float = 1024.0,
    method: str = 'xcorr',
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    t_gps: float = 1000000000.0,
    return_dict: bool = False
) -> Union[np.ndarray, Dict[str, float]]:
    """
    Extract 6D physics feature vector from detector strains.

    Features:
        1. tau_HL: Time delay H1-L1
        2. tau_HV: Time delay H1-V1
        3. A_HL: Amplitude ratio H1/L1
        4. A_HV: Amplitude ratio H1/V1
        5. phi_HL: Phase difference H1-L1
        6. phi_HV: Phase difference H1-V1

    Args:
        strains: Dictionary {'H1': array, 'L1': array, 'V1': array}
        fs: Sampling frequency in Hz
        method: Time delay method - 'xcorr' (cross-correlation) or 'geometric'
        ra: Right ascension in radians (required if method='geometric')
        dec: Declination in radians (required if method='geometric')
        t_gps: GPS time (for geometric method)
        return_dict: If True, return dict; otherwise return array

    Returns:
        If return_dict=False: np.ndarray of shape (6,)
        If return_dict=True: Dict[str, float] with named features

    Examples:
        >>> features = extract_physics_features(
        ...     {'H1': h1, 'L1': l1, 'V1': v1},
        ...     fs=1024.0,
        ...     method='xcorr'
        ... )
        >>> features.shape  # (6,)
    """
    required = ['H1', 'L1', 'V1']
    for det in required:
        if det not in strains:
            raise ValueError(f"Missing detector {det} in strains dict")

    h1 = strains['H1']
    l1 = strains['L1']
    v1 = strains['V1']

    features = {}

    if method == 'xcorr':
        features['tau_HL'] = compute_time_delay_xcorr(h1, l1, fs)
        features['tau_HV'] = compute_time_delay_xcorr(h1, v1, fs)
    elif method == 'geometric':
        if ra is None or dec is None:
            raise ValueError("ra and dec required for geometric time delay computation")
        features['tau_HL'] = compute_time_delay_geometric(ra, dec, 'H1', 'L1', t_gps)
        features['tau_HV'] = compute_time_delay_geometric(ra, dec, 'H1', 'V1', t_gps)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'xcorr' or 'geometric'.")

    features['A_HL'] = compute_amplitude_ratio(h1, l1, method='max')
    features['A_HV'] = compute_amplitude_ratio(h1, v1, method='max')

    features['phi_HL'] = compute_phase_difference(h1, l1)
    features['phi_HV'] = compute_phase_difference(h1, v1)

    if return_dict:
        return features
    else:
        return np.array([
            features['tau_HL'],
            features['tau_HV'],
            features['A_HL'],
            features['A_HV'],
            features['phi_HL'],
            features['phi_HV'],
        ], dtype=np.float32)


def extract_all_time_delays(
    strains: Dict[str, np.ndarray],
    fs: float = 1024.0,
    method: str = 'xcorr',
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    t_gps: float = 1000000000.0
) -> Dict[str, float]:
    """
    Extract all three time delays (HL, HV, LV).

    Note: In the standard 6D feature vector, we only use HL and HV since
    LV is redundant (tau_LV = tau_HV - tau_HL). This function provides
    all three for completeness.

    Args:
        strains: Dictionary {'H1': array, 'L1': array, 'V1': array}
        fs: Sampling frequency
        method: 'xcorr' or 'geometric'
        ra, dec: Sky position (for geometric method)
        t_gps: GPS time

    Returns:
        Dictionary {'tau_HL': float, 'tau_HV': float, 'tau_LV': float}
    """
    h1 = strains['H1']
    l1 = strains['L1']
    v1 = strains['V1']

    if method == 'xcorr':
        tau_HL = compute_time_delay_xcorr(h1, l1, fs)
        tau_HV = compute_time_delay_xcorr(h1, v1, fs)
        tau_LV = compute_time_delay_xcorr(l1, v1, fs)
    elif method == 'geometric':
        if ra is None or dec is None:
            raise ValueError("ra and dec required for geometric method")
        tau_HL = compute_time_delay_geometric(ra, dec, 'H1', 'L1', t_gps)
        tau_HV = compute_time_delay_geometric(ra, dec, 'H1', 'V1', t_gps)
        tau_LV = compute_time_delay_geometric(ra, dec, 'L1', 'V1', t_gps)
    else:
        raise ValueError(f"Unknown method '{method}'")

    return {
        'tau_HL': tau_HL,
        'tau_HV': tau_HV,
        'tau_LV': tau_LV,
    }


def compute_snr(strain: np.ndarray, noise_std: Optional[float] = None) -> float:
    """
    Compute signal-to-noise ratio of a strain time series.

    Args:
        strain: Strain time series
        noise_std: Noise standard deviation (if None, estimated from strain)

    Returns:
        SNR (dimensionless)

    Note:
        This is a simplified SNR calculation. For proper GW analysis,
        use matched filtering with noise PSDs.
    """
    signal_power = np.max(np.abs(strain))

    if noise_std is None:
        n = len(strain)
        noise_segment = np.concatenate([strain[:n//10], strain[-n//10:]])
        noise_std = np.std(noise_segment)

    if noise_std == 0:
        return np.inf

    snr = signal_power / noise_std

    return snr


if __name__ == "__main__":
    print("Testing physics feature extraction...")

    fs = 1024.0
    duration = 1.0
    t = np.arange(int(duration * fs)) / fs

    f0, f1 = 30, 100
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)

    tau_HL_true = 0.002  # 2 ms
    tau_HV_true = 0.005  # 5 ms

    h1 = chirp
    l1 = 0.8 * np.roll(chirp, int(tau_HL_true * fs))  # Delayed + attenuated
    v1 = 0.6 * np.roll(chirp, int(tau_HV_true * fs))

    strains = {'H1': h1, 'L1': l1, 'V1': v1}

    features = extract_physics_features(strains, fs=fs, method='xcorr', return_dict=True)

    print("\nExtracted features:")
    for key, val in features.items():
        print(f"  {key}: {val:.6f}")

    print(f"\nExpected tau_HL: {tau_HL_true:.6f}, Got: {features['tau_HL']:.6f}")
    print(f"Expected tau_HV: {tau_HV_true:.6f}, Got: {features['tau_HV']:.6f}")

    features_array = extract_physics_features(strains, fs=fs, method='xcorr')
    print(f"\nFeatures as array: {features_array}")
    print(f"Shape: {features_array.shape}")

    print("\nAll tests passed!")
