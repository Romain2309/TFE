"""
PyTorch Dataset and DataLoader utilities for GW sky localization.

Provides:
- GWDataset: PyTorch Dataset wrapping PreGeneratedDataReader
- create_dataloaders: Factory function for train/val/test splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from .file_readers import PreGeneratedDataReader
from ..utils.coordinates import radec_to_healpix, get_healpix_npix


class GWDataset(Dataset):
    """
    PyTorch Dataset for gravitational wave events.

    Wraps PreGeneratedDataReader and provides PyTorch-compatible output.
    """

    def __init__(
        self,
        data_dir: str,
        target_length: int = 2048,
        bandpass: Optional[Tuple[float, float]] = (30, 1024),
        normalize: bool = True,
        use_healpix: bool = False,
        healpix_nside: int = 8,
        cache: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing .npy files
            target_length: Target sequence length
            bandpass: Bandpass filter (low, high) or None
            normalize: Whether to normalize strains
            use_healpix: Whether to compute HEALPix labels
            healpix_nside: HEALPix resolution parameter
            cache: Whether to cache loaded events in memory (use for small datasets)
            verbose: Print progress information
        """
        self.reader = PreGeneratedDataReader(
            data_dir=data_dir,
            target_length=target_length,
            bandpass=bandpass,
            normalize=normalize,
            verbose=verbose,
        )

        self.use_healpix = use_healpix
        self.healpix_nside = healpix_nside

        if self.use_healpix:
            self.n_pixels = get_healpix_npix(healpix_nside)

        self.cache = cache
        self._cache_dict = {} if cache else None

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single event as PyTorch tensors.

        Returns:
            Dictionary with:
                - strain: (3, seq_len) FloatTensor
                - physics_features: (6,) FloatTensor
                - xyz: (3,) FloatTensor (unit vector target)
                - ra: FloatTensor (scalar, radians)
                - dec: FloatTensor (scalar, radians)
                - healpix_label: LongTensor (scalar, pixel index) [if use_healpix=True]
                - event_id: LongTensor (scalar)
        """
        if self.cache and idx in self._cache_dict:
            return self._cache_dict[idx]

        event = self.reader[idx]

        item = {
            'strain': torch.from_numpy(event['strain']),
            'physics_features': torch.from_numpy(event['physics_features']),
            'xyz': torch.from_numpy(event['xyz']),
            'ra': torch.tensor(event['ra'], dtype=torch.float32),
            'dec': torch.tensor(event['dec'], dtype=torch.float32),
            'event_id': torch.tensor(event['event_id'], dtype=torch.long),
        }

        if self.use_healpix:
            healpix_label = radec_to_healpix(event['ra'], event['dec'], nside=self.healpix_nside)
            item['healpix_label'] = torch.tensor(healpix_label, dtype=torch.long)

        if self.cache:
            self._cache_dict[idx] = item

        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])

    return collated


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    target_length: int = 2048,
    bandpass: Optional[Tuple[float, float]] = (30, 1024),
    normalize: bool = True,
    use_healpix: bool = False,
    healpix_nside: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    max_events: Optional[int] = None,
    cache_train: bool = False,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir: Directory containing .npy files
        batch_size: Batch size for all dataloaders
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
        target_length: Target sequence length
        bandpass: Bandpass filter frequencies
        normalize: Whether to normalize strains
        use_healpix: Whether to compute HEALPix labels
        healpix_nside: HEALPix resolution
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducible splits
        max_events: Maximum number of events to use (for quick testing)
        cache_train: Whether to cache training data in memory
        verbose: Print information

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = GWDataset(
        data_dir=data_dir,
        target_length=target_length,
        bandpass=bandpass,
        normalize=normalize,
        use_healpix=use_healpix,
        healpix_nside=healpix_nside,
        cache=False,  # Don't cache full dataset
        verbose=verbose,
    )

    n_total = len(dataset)

    if max_events is not None:
        n_total = min(n_total, max_events)

    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    if verbose:

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    if cache_train:
        if verbose:
        train_dataset_cached = GWDataset(
            data_dir=data_dir,
            target_length=target_length,
            bandpass=bandpass,
            normalize=normalize,
            use_healpix=use_healpix,
            healpix_nside=healpix_nside,
            cache=True,
            verbose=False,
        )
        train_dataset = Subset(train_dataset_cached, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    data_dir = '/data/stu_bonhomme/tfe/benchmark_same_small'

    dataset = GWDataset(
        data_dir=data_dir,
        use_healpix=True,
        healpix_nside=8,
        verbose=True
    )


    sample = dataset[0]
    for key, val in sample.items():

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        max_events=100,  # Small test
        num_workers=2,
        use_healpix=True,
        verbose=True,
    )

    batch = next(iter(train_loader))
    for key, val in batch.items():

