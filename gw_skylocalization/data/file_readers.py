"""
File readers for pre-generated gravitational wave data.

Handles loading .npy files with the format:
    EVENT_{id}_CONFIG_{cfg}_{detector}_RA_{ra}_DEC_{dec}_{hash}.npy
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from ..utils.physics import extract_physics_features
from ..utils.coordinates import radec_to_xyz, radec_to_healpix


class PreGeneratedDataReader:
    """
    Reader for pre-generated gravitational wave event data.

    Scans a directory for .npy files following the naming convention:
        EVENT_{id}_CONFIG_{cfg}_{detector}_RA_{ra}_DEC_{dec}_{hash}.npy

    Groups files by event and provides preprocessing functionality.
    """

    def __init__(
        self,
        data_dir: str,
        target_length: int = 2048,
        bandpass: Optional[Tuple[float, float]] = (30, 1024),
        normalize: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing .npy files
            target_length: Target sequence length (crop/pad)
            bandpass: Bandpass filter frequencies (low, high) or None
            normalize: Whether to z-score normalize each detector
            verbose: Print progress information
        """
        self.data_dir = Path(data_dir)
        self.target_length = target_length
        self.bandpass = bandpass
        self.normalize = normalize
        self.verbose = verbose

        self.pattern1 = re.compile(
            r'EVENT_(\d+)_CONFIG_(\d+)_([HLV]1)_RA_([\d.]+)_DEC_([-\d.]+)_([A-Za-z0-9]+)\.npy'
        )
        self.pattern2 = re.compile(
            r'EVENT_ID_(\d+)_([HLV]1)_.*?_RA-([\d.]+)_DEC-([-\d.]+)\.npy'
        )

        self.events = self._scan_directory()

        if self.verbose:
            print(f"PreGeneratedDataReader initialized:")
            print(f"  Data directory: {self.data_dir}")
            print(f"  Complete events found: {len(self.events)}")
            print(f"  Target length: {self.target_length}")
            print(f"  Bandpass: {self.bandpass}")
            print(f"  Normalize: {self.normalize}")

    def _scan_directory(self) -> List[Dict]:
        """Scan directory and group files by event."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        events = defaultdict(dict)

        for filepath in self.data_dir.glob('*.npy'):
            match = self.pattern1.match(filepath.name)
            if match:
                event_id = int(match.group(1))
                config_id = int(match.group(2))
                detector = match.group(3)
                ra = float(match.group(4))
                dec = float(match.group(5))
                hash_id = match.group(6)

                key = (event_id, config_id, hash_id)

                if key not in events:
                    events[key] = {
                        'event_id': event_id,
                        'config_id': config_id,
                        'ra': ra,
                        'dec': dec,
                        'hash': hash_id,
                        'files': {}
                    }

                events[key]['files'][detector] = str(filepath)
            else:
                match = self.pattern2.match(filepath.name)
                if match:
                    event_id = int(match.group(1))
                    detector = match.group(2)
                    ra = float(match.group(3))
                    dec = float(match.group(4))

                    key = (event_id, 0, 'advanced')

                    if key not in events:
                        events[key] = {
                            'event_id': event_id,
                            'config_id': 0,
                            'ra': ra,
                            'dec': dec,
                            'hash': 'advanced',
                            'files': {}
                        }

                    events[key]['files'][detector] = str(filepath)

        complete_events = [
            event for event in events.values()
            if len(event['files']) == 3 and all(det in event['files'] for det in ['H1', 'L1', 'V1'])
        ]

        complete_events.sort(key=lambda x: (x['event_id'], x['config_id']))

        return complete_events

    def _apply_bandpass(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Apply bandpass filter to signal."""
        if self.bandpass is None:
            return signal

        nyquist = fs / 2
        low, high = self.bandpass
        low_norm = low / nyquist
        high_norm = high / nyquist

        low_norm = np.clip(low_norm, 0.001, 0.999)
        high_norm = np.clip(high_norm, 0.001, 0.999)

        if low_norm >= high_norm:
            raise ValueError(f"Invalid bandpass: {self.bandpass} at fs={fs}")

        b, a = butter(4, [low_norm, high_norm], btype='band')
        filtered = filtfilt(b, a, signal)

        return filtered.astype(np.float32)

    def _crop_or_pad(self, signal: np.ndarray) -> np.ndarray:
        """Crop or pad signal to target length."""
        if len(signal) > self.target_length:
            start = (len(signal) - self.target_length) // 2
            signal = signal[start:start + self.target_length]
        elif len(signal) < self.target_length:
            pad_left = (self.target_length - len(signal)) // 2
            pad_right = self.target_length - len(signal) - pad_left
            signal = np.pad(signal, (pad_left, pad_right), mode='constant')

        return signal

    def load_event(self, idx: int) -> Dict:
        """
        Load and preprocess a single event.

        Args:
            idx: Event index (0 to len(self.events)-1)

        Returns:
            Dictionary with:
                - strain: (3, target_length) array
                - physics_features: (6,) array
                - xyz: (3,) unit vector
                - ra: float (radians)
                - dec: float (radians)
                - event_id: int
                - config_id: int
                - fs: float (sampling rate)
        """
        event = self.events[idx]

        strains_raw = {}
        fs = None

        for det in ['H1', 'L1', 'V1']:
            data = np.load(event['files'][det], allow_pickle=True).item()
            timeseries = data[det]
            strains_raw[det] = np.asarray(timeseries.value, dtype=np.float32)

            if fs is None:
                fs = float(timeseries.sample_rate.value)

        min_len = min(len(strains_raw[det]) for det in ['H1', 'L1', 'V1'])
        for det in ['H1', 'L1', 'V1']:
            strains_raw[det] = strains_raw[det][:min_len]

        strains_filtered = {}
        for det in ['H1', 'L1', 'V1']:
            strains_filtered[det] = self._apply_bandpass(strains_raw[det], fs)

        if self.normalize:
            for det in ['H1', 'L1', 'V1']:
                mean = np.mean(strains_filtered[det])
                std = np.std(strains_filtered[det])
                if std > 0:
                    strains_filtered[det] = (strains_filtered[det] - mean) / std

        strains_processed = {}
        for det in ['H1', 'L1', 'V1']:
            strains_processed[det] = self._crop_or_pad(strains_filtered[det])

        strain_array = np.stack([
            strains_processed['H1'],
            strains_processed['L1'],
            strains_processed['V1'],
        ], axis=0).astype(np.float32)

        ra_rad = event['ra']
        dec_rad = event['dec']

        physics_features = extract_physics_features(
            strains_processed,
            fs=fs,
            method='geometric',  # Changed from 'xcorr' to 'geometric'
            ra=ra_rad,
            dec=dec_rad,
            return_dict=False
        )
        xyz = radec_to_xyz(ra_rad, dec_rad)

        return {
            'strain': strain_array,
            'physics_features': physics_features,
            'xyz': xyz,
            'ra': ra_rad,
            'dec': dec_rad,
            'event_id': event['event_id'],
            'config_id': event['config_id'],
            'fs': fs,
        }

    def load_batch(
        self,
        indices: List[int],
        show_progress: bool = False
    ) -> List[Dict]:
        """
        Load multiple events.

        Args:
            indices: List of event indices
            show_progress: Show tqdm progress bar

        Returns:
            List of event dictionaries
        """
        iterator = tqdm(indices, desc="Loading events") if show_progress else indices
        batch = []

        for idx in iterator:
            try:
                event_data = self.load_event(idx)
                batch.append(event_data)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading event {idx}: {e}")
                continue

        return batch

    def __len__(self) -> int:
        """Number of complete events."""
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict:
        """Get event by index (alias for load_event)."""
        return self.load_event(idx)


def collate_events(batch: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Collate a batch of events into arrays.

    Args:
        batch: List of event dictionaries from PreGeneratedDataReader

    Returns:
        Dictionary with batched arrays:
            - strain: (batch, 3, seq_len)
            - physics_features: (batch, 6)
            - xyz: (batch, 3)
            - ra: (batch,)
            - dec: (batch,)
            - event_ids: (batch,)
            - config_ids: (batch,)
    """
    return {
        'strain': np.stack([e['strain'] for e in batch]),
        'physics_features': np.stack([e['physics_features'] for e in batch]),
        'xyz': np.stack([e['xyz'] for e in batch]),
        'ra': np.array([e['ra'] for e in batch]),
        'dec': np.array([e['dec'] for e in batch]),
        'event_ids': np.array([e['event_id'] for e in batch]),
        'config_ids': np.array([e['config_id'] for e in batch]),
    }


"""
if __name__ == "__main__":
    print("Testing PreGeneratedDataReader...")

    data_dir = '/data/stu_bonhomme/tfe/benchmark_same_small'

    reader = PreGeneratedDataReader(data_dir, verbose=True)

    print(f"\nTotal events: {len(reader)}")

    print("\nLoading first event...")
    event = reader[0]

    print(f"Event ID: {event['event_id']}")
    print(f"RA: {np.degrees(event['ra']):.2f}°, Dec: {np.degrees(event['dec']):.2f}°")
    print(f"Strain shape: {event['strain'].shape}")
    print(f"Physics features shape: {event['physics_features'].shape}")
    print(f"Physics features: {event['physics_features']}")
    print(f"Unit vector (xyz): {event['xyz']}")
    print(f"Sampling rate: {event['fs']} Hz")

    print("\nLoading batch of 10 events...")
    batch = reader.load_batch(list(range(10)), show_progress=True)
    print(f"Loaded {len(batch)} events")

    print("\nCollating batch...")
    collated = collate_events(batch)
    for key, val in collated.items():
        print(f"  {key}: {val.shape}")

    print("\nTest passed!")
"""
