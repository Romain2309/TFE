"""
Input/Output utilities for saving and loading models, datasets, and results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import h5py


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    save_path: str,
    **extra_state
):
    """
    Save model checkpoint with full training state.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (can be None)
        epoch: Current epoch number
        metrics: Dictionary of metrics
        config: Training configuration
        save_path: Path to save checkpoint
        **extra_state: Additional state to save (e.g., scheduler, scaler)

    Saved format:
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'optimizer_state_dict': OrderedDict,
            'metrics': dict,
            'config': dict,
            **extra_state
        }
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    checkpoint.update(extra_state)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into (if None, only return checkpoint)
        optimizer: PyTorch optimizer to load state into (optional)
        device: Device to map model to

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")

    return checkpoint


def save_model(model: torch.nn.Module, save_path: str):
    """
    Save only model weights (not full checkpoint).

    Args:
        model: PyTorch model
        save_path: Path to save weights
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def load_model(
    model: torch.nn.Module,
    weights_path: str,
    device: str = 'cpu',
    strict: bool = True
):
    """
    Load model weights only.

    Args:
        model: PyTorch model
        weights_path: Path to weights file
        device: Device to map weights to
        strict: Whether to strictly enforce key matching
    """
    state_dict = torch.load(weights_path, map_location=device)

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=strict)
    print(f"Model weights loaded from {weights_path}")


def save_dataset_h5(
    save_path: str,
    strains: np.ndarray,
    physics_features: np.ndarray,
    xyz: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    healpix_labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = 'gzip'
):
    """
    Save dataset to HDF5 file with compression.

    Args:
        save_path: Path to save HDF5 file
        strains: Strain data, shape (n_events, 3, seq_len)
        physics_features: Physics features, shape (n_events, 6)
        xyz: Unit vectors, shape (n_events, 3)
        ra: Right ascension, shape (n_events,)
        dec: Declination, shape (n_events,)
        healpix_labels: HEALPix pixel labels, shape (n_events,) (optional)
        metadata: Additional metadata dictionary
        compression: Compression algorithm ('gzip', 'lzf', None)

    HDF5 structure:
        /strains             - (n, 3, seq_len) float32
        /physics_features    - (n, 6) float32
        /xyz                 - (n, 3) float32
        /ra                  - (n,) float32
        /dec                 - (n,) float32
        /healpix_labels      - (n,) int32 (optional)
        /metadata (attrs)    - dict
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('strains', data=strains, compression=compression, dtype='float32')
        f.create_dataset('physics_features', data=physics_features, compression=compression, dtype='float32')
        f.create_dataset('xyz', data=xyz, compression=compression, dtype='float32')
        f.create_dataset('ra', data=ra, compression=compression, dtype='float32')
        f.create_dataset('dec', data=dec, compression=compression, dtype='float32')

        if healpix_labels is not None:
            f.create_dataset('healpix_labels', data=healpix_labels, compression=compression, dtype='int32')

        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value

    print(f"Dataset saved to {save_path} ({save_path.stat().st_size / 1e6:.2f} MB)")


def load_dataset_h5(
    load_path: str,
    load_metadata: bool = True
) -> Dict[str, Any]:
    """
    Load dataset from HDF5 file.

    Args:
        load_path: Path to HDF5 file
        load_metadata: Whether to load metadata attributes

    Returns:
        Dictionary containing:
            - strains: (n, 3, seq_len)
            - physics_features: (n, 6)
            - xyz: (n, 3)
            - ra: (n,)
            - dec: (n,)
            - healpix_labels: (n,) (if present)
            - metadata: dict (if load_metadata=True)
    """
    data = {}

    with h5py.File(load_path, 'r') as f:
        data['strains'] = f['strains'][:]
        data['physics_features'] = f['physics_features'][:]
        data['xyz'] = f['xyz'][:]
        data['ra'] = f['ra'][:]
        data['dec'] = f['dec'][:]

        if 'healpix_labels' in f:
            data['healpix_labels'] = f['healpix_labels'][:]

        if load_metadata:
            data['metadata'] = dict(f.attrs)

    print(f"Dataset loaded from {load_path} ({len(data['ra'])} events)")

    return data


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = 'results'
):
    """
    Save evaluation results to JSON and NPZ files.

    Args:
        results: Dictionary of results (metrics + arrays)
        output_dir: Directory to save results
        prefix: Filename prefix

    Saves:
        - {prefix}_metrics.json: Scalar metrics
        - {prefix}_arrays.npz: Array data (predictions, errors, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    arrays = {}

    for key, value in results.items():
        if isinstance(value, (np.ndarray, list)):
            arrays[key] = np.asarray(value)
        elif isinstance(value, (int, float, str, bool)):
            metrics[key] = value
        elif isinstance(value, dict):
            metrics[key] = value
        else:
            try:
                arrays[key] = np.asarray(value)
            except:
                metrics[key] = str(value)

    metrics_path = output_dir / f'{prefix}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if arrays:
        arrays_path = output_dir / f'{prefix}_arrays.npz'
        np.savez_compressed(arrays_path, **arrays)
        print(f"Arrays saved to {arrays_path}")


def load_results(
    results_dir: str,
    prefix: str = 'results'
) -> Dict[str, Any]:
    """
    Load results from JSON and NPZ files.

    Args:
        results_dir: Directory containing results
        prefix: Filename prefix

    Returns:
        Combined dictionary of metrics and arrays
    """
    results_dir = Path(results_dir)
    results = {}

    metrics_path = results_dir / f'{prefix}_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results.update(json.load(f))

    arrays_path = results_dir / f'{prefix}_arrays.npz'
    if arrays_path.exists():
        with np.load(arrays_path) as data:
            results.update(dict(data))

    return results


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


"""
if __name__ == "__main__":
    print("Testing I/O utilities...")

    n_events = 10
    seq_len = 2048

    fake_data = {
        'strains': np.random.randn(n_events, 3, seq_len).astype(np.float32),
        'physics_features': np.random.randn(n_events, 6).astype(np.float32),
        'xyz': np.random.randn(n_events, 3).astype(np.float32),
        'ra': np.random.uniform(0, 2*np.pi, n_events).astype(np.float32),
        'dec': np.random.uniform(-np.pi/2, np.pi/2, n_events).astype(np.float32),
    }

    save_dataset_h5(
        'test_dataset.h5',
        metadata={'n_events': n_events, 'fs': 1024},
        **fake_data
    )

    loaded = load_dataset_h5('test_dataset.h5')

    assert np.allclose(fake_data['strains'], loaded['strains'])
    assert np.allclose(fake_data['ra'], loaded['ra'])
    assert loaded['metadata']['n_events'] == n_events

    print("HDF5 save/load test passed!")

    Path('test_dataset.h5').unlink()

    print("\nAll tests passed!")
"""
