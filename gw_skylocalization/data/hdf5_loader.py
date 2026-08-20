"""
Fast HDF5 data loader for preprocessed datasets.

Loads preprocessed data from HDF5 file created by scripts/preprocess_dataset.py
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, Tuple, Optional
from pathlib import Path


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for preprocessed HDF5 data.

    Ultra-fast loading - no preprocessing during training!
    """

    def __init__(
        self,
        hdf5_path: str,
        in_memory: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            hdf5_path: Path to preprocessed HDF5 file
            in_memory: Load entire dataset into RAM (for small datasets)
            normalize: Apply dataset-level normalization
        """
        self.hdf5_path = Path(hdf5_path)
        self.in_memory = in_memory
        self.normalize = normalize

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_events = f['strain'].shape[0]
            self.target_length = f.attrs.get('target_length', 2048)
            self.bandpass = (
                f.attrs.get('bandpass_low', 30.0),
                f.attrs.get('bandpass_high', 1024.0)
            )

        self.strain_mean = None
        self.strain_std = None
        self.physics_mean = None
        self.physics_std = None

        if in_memory:
            print(f"Loading dataset into memory ({self.n_events} events)...")
            with h5py.File(self.hdf5_path, 'r') as f:
                self.strain = f['strain'][:]
                self.physics_features = f['physics_features'][:]
                self.xyz = f['xyz'][:]
                self.healpix_label = f['healpix_label'][:]
            print("✅ Dataset loaded into memory!")
        else:
            self.strain = None
            self.physics_features = None
            self.xyz = None
            self.healpix_label = None
            self._file = None

    def _ensure_file_open(self):
        """Ensure HDF5 file is open (for multi-worker compatibility)."""
        if not self.in_memory and self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')

    def __len__(self) -> int:
        return self.n_events

    def compute_normalization_stats(self, indices: Optional[np.ndarray] = None):
        """
        Compute dataset-level normalization statistics.

        Args:
            indices: Indices to compute stats from (e.g., training set only)
                    If None, uses all events
        """

        if self.in_memory:
            if indices is None:
                strains = self.strain
                physics = self.physics_features
            else:
                strains = self.strain[indices]
                physics = self.physics_features[indices]

            self.strain_mean = strains.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            self.strain_std = strains.std(axis=(0, 2), keepdims=True).astype(np.float32)

            self.physics_mean = physics.mean(axis=0).astype(np.float32)
            self.physics_std = physics.std(axis=0).astype(np.float32)
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                if indices is not None:
                    sorted_indices = np.sort(indices)
                    n_samples = len(sorted_indices)
                else:
                    n_samples = f['strain'].shape[0]
                    sorted_indices = np.arange(n_samples)

                batch_size = min(1000, n_samples)

                n_channels = 3
                seq_len = f['strain'].shape[2]
                n_physics = 6

                strain_sum = np.zeros((1, n_channels, 1), dtype=np.float64)
                strain_sq_sum = np.zeros((1, n_channels, 1), dtype=np.float64)
                strain_count = 0

                physics_sum = np.zeros(n_physics, dtype=np.float64)
                physics_sq_sum = np.zeros(n_physics, dtype=np.float64)
                physics_count = 0

                for i in range(0, n_samples, batch_size):
                    batch_idx = sorted_indices[i:i+batch_size].tolist()

                    batch_strains = f['strain'][batch_idx]
                    batch_physics = f['physics_features'][batch_idx]

                    strain_sum += batch_strains.sum(axis=(0, 2), keepdims=True)
                    strain_sq_sum += (batch_strains ** 2).sum(axis=(0, 2), keepdims=True)
                    strain_count += batch_strains.shape[0] * batch_strains.shape[2]

                    physics_sum += batch_physics.sum(axis=0)
                    physics_sq_sum += (batch_physics ** 2).sum(axis=0)
                    physics_count += batch_physics.shape[0]

                self.strain_mean = (strain_sum / strain_count).astype(np.float32)
                self.strain_std = np.sqrt(strain_sq_sum / strain_count - self.strain_mean ** 2).astype(np.float32)

                self.physics_mean = (physics_sum / physics_count).astype(np.float32)
                self.physics_std = np.sqrt(physics_sq_sum / physics_count - self.physics_mean ** 2).astype(np.float32)

        print(f"  Strain mean: {self.strain_mean.flatten()}")
        print(f"  Strain std: {self.strain_std.flatten()}")
        print(f"  Physics mean: {self.physics_mean}")
        print(f"  Physics std: {self.physics_std}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single event.

        Returns:
            Dictionary with:
            - strain: (3, seq_len) tensor
            - physics_features: (6,) tensor
            - xyz: (3,) tensor
            - healpix_label: scalar tensor
        """
        if self.in_memory:
            strain = self.strain[idx].copy()
            physics_features = self.physics_features[idx].copy()
            xyz = self.xyz[idx]
            healpix_label = self.healpix_label[idx]
        else:
            self._ensure_file_open()
            strain = self._file['strain'][idx]
            physics_features = self._file['physics_features'][idx]
            xyz = self._file['xyz'][idx]
            healpix_label = self._file['healpix_label'][idx]

        if self.normalize:
            if self.strain_mean is not None and self.strain_std is not None:
                strain = (strain - self.strain_mean.squeeze(0)) / self.strain_std.squeeze(0)
            if self.physics_mean is not None and self.physics_std is not None:
                physics_features = (physics_features - self.physics_mean) / self.physics_std

        return {
            'strain': torch.from_numpy(strain).float(),
            'physics_features': torch.from_numpy(physics_features).float(),
            'xyz': torch.from_numpy(xyz).float(),
            'healpix_label': torch.tensor(healpix_label, dtype=torch.long),
        }

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self._file is not None:
            self._file.close()


def create_hdf5_dataloaders(
    hdf5_path: str,
    batch_size: int = 32,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    in_memory: bool = False,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from HDF5 file.

    Args:
        hdf5_path: Path to preprocessed HDF5 file
        batch_size: Batch size
        train_frac: Training fraction
        val_frac: Validation fraction
        test_frac: Test fraction
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for splits
        in_memory: Load entire dataset into RAM
        verbose: Print info

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = HDF5Dataset(hdf5_path, in_memory=in_memory, normalize=True)

    n_events = len(dataset)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_events)

    n_train = int(n_events * train_frac)
    n_val = int(n_events * val_frac)
    n_test = n_events - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    if verbose:
        print(f"Dataset splits:")
        print(f"  Total events: {n_events}")
        print(f"  Train: {n_train} ({train_frac*100:.1f}%)")
        print(f"  Val: {n_val} ({val_frac*100:.1f}%)")
        print(f"  Test: {n_test} ({test_frac*100:.1f}%)")

    if verbose:
        print("\nComputing normalization statistics on training set...")
    dataset.compute_normalization_stats(train_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    def collate_fn(batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    if verbose:
        print(f"  Train: {len(train_loader.dataset)} events ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_loader.dataset)} events ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_loader.dataset)} events ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader


"""
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    else:
        hdf5_path = '/data/stu_bonhomme/tfe/dataset/romain_preprocessed/preprocessed_data.h5'

    print(f"Testing HDF5Dataset with {hdf5_path}")

    dataset = HDF5Dataset(hdf5_path)
    print(f"  Events: {len(dataset)}")

    sample = dataset[0]
    print(f"  Strain shape: {sample['strain'].shape}")
    print(f"  Physics shape: {sample['physics_features'].shape}")
    print(f"  XYZ shape: {sample['xyz'].shape}")
    print(f"  HEALPix label: {sample['healpix_label']}")
"""
