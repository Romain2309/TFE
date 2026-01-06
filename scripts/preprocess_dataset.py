#!/usr/bin/env python3
"""
Preprocess dataset and save to HDF5 for fast loading.

This script:
1. Loads all events from pre-generated .npy files
2. Applies bandpass filtering (30-1024 Hz)
3. Normalizes strains
4. Extracts physics features
5. Saves to HDF5 file for instant loading during training

Usage:
    python scripts/preprocess_dataset.py \
        --input-dir /data/stu_bonhomme/tfe/benchmark_same_small \
        --output-dir /data/stu_bonhomme/tfe/dataset/romain_preprocessed
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gw_skylocalization.data.file_readers import PreGeneratedDataReader
from gw_skylocalization.utils.coordinates import radec_to_healpix


def preprocess_and_save(
    input_dir: str,
    output_dir: str,
    target_length: int = 2048,
    bandpass_low: float = 30.0,
    bandpass_high: float = 1024.0,
    normalize: bool = True,
    healpix_nside: int = 8,
):
    """
    Preprocess all events and save to HDF5.

    Args:
        input_dir: Directory with .npy files
        output_dir: Output directory for HDF5 file
        target_length: Target sequence length
        bandpass_low: Low frequency cutoff
        bandpass_high: High frequency cutoff
        normalize: Whether to normalize
        healpix_nside: HEALPix nside parameter
    """
    print("="*80)
    print("Romain Dataset Preprocessing")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target length: {target_length}")
    print(f"Bandpass: ({bandpass_low}, {bandpass_high}) Hz")
    print(f"Normalize: {normalize}")
    print(f"HEALPix nside: {healpix_nside}")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize reader
    print("\nInitializing data reader...")
    reader = PreGeneratedDataReader(
        data_dir=input_dir,
        target_length=target_length,
        bandpass=(bandpass_low, bandpass_high),
        normalize=normalize,
        verbose=True,
    )

    n_events = len(reader)
    print(f"\nFound {n_events} complete events")

    # Prepare HDF5 file
    output_file = output_path / 'preprocessed_data.h5'
    print(f"\nCreating HDF5 file: {output_file}")
    print("DEBUG: About to open HDF5 file...")
    import sys
    sys.stdout.flush()

    with h5py.File(output_file, 'w') as f:
        print("DEBUG: HDF5 file opened successfully!")
        sys.stdout.flush()

        # Create datasets
        print("Creating datasets (this may take a minute)...")
        sys.stdout.flush()

        # Preallocate arrays (no compression for speed)
        print("  DEBUG: About to create strain dataset...")
        sys.stdout.flush()

        # Calculate appropriate chunk size (must be < 4GB)
        # chunk_size = chunk_events * 3 * target_length * 4 bytes
        max_chunk_bytes = 3.5 * 1024**3  # 3.5 GB to be safe
        bytes_per_event = 3 * target_length * 4
        chunk_events = min(100, int(max_chunk_bytes / bytes_per_event))
        chunk_events = max(1, chunk_events)  # At least 1

        print(f"  DEBUG: Using chunk size: ({chunk_events}, 3, {target_length})")
        sys.stdout.flush()

        strain_ds = f.create_dataset(
            'strain',
            shape=(n_events, 3, target_length),
            dtype=np.float32,
            chunks=(chunk_events, 3, target_length),  # Chunked for better I/O
        )
        print("  DEBUG: Strain dataset created!")
        sys.stdout.flush()

        print("  Creating physics features dataset...")
        sys.stdout.flush()
        physics_ds = f.create_dataset(
            'physics_features',
            shape=(n_events, 6),
            dtype=np.float32,
        )
        print("  DEBUG: Physics features created!")
        sys.stdout.flush()

        print("  Creating xyz dataset...")
        sys.stdout.flush()
        xyz_ds = f.create_dataset(
            'xyz',
            shape=(n_events, 3),
            dtype=np.float32,
        )
        print("  DEBUG: XYZ created!")
        sys.stdout.flush()

        print("  Creating ra dataset...")
        sys.stdout.flush()
        ra_ds = f.create_dataset(
            'ra',
            shape=(n_events,),
            dtype=np.float32,
        )
        print("  DEBUG: RA created!")
        sys.stdout.flush()

        print("  Creating dec dataset...")
        sys.stdout.flush()
        dec_ds = f.create_dataset(
            'dec',
            shape=(n_events,),
            dtype=np.float32,
        )
        print("  DEBUG: DEC created!")
        sys.stdout.flush()

        print("  Creating healpix_label dataset...")
        sys.stdout.flush()
        healpix_ds = f.create_dataset(
            'healpix_label',
            shape=(n_events,),
            dtype=np.int32,
        )
        print("  DEBUG: HEALPix created!")
        sys.stdout.flush()

        print("✅ Datasets created!")
        sys.stdout.flush()

        # Store metadata
        print("DEBUG: Storing metadata...")
        sys.stdout.flush()
        f.attrs['n_events'] = n_events
        f.attrs['target_length'] = target_length
        f.attrs['bandpass_low'] = bandpass_low
        f.attrs['bandpass_high'] = bandpass_high
        f.attrs['normalize'] = normalize
        f.attrs['healpix_nside'] = healpix_nside
        print("DEBUG: Metadata stored!")
        sys.stdout.flush()

        # Process and save all events
        print("\nDEBUG: About to start processing loop...")
        sys.stdout.flush()
        print("Processing events...")
        sys.stdout.flush()
        for idx in tqdm(range(n_events), desc="Preprocessing"):
            event = reader.load_event(idx)

            # Compute HEALPix label from ra/dec
            healpix_label = radec_to_healpix(event['ra'], event['dec'], nside=healpix_nside)

            strain_ds[idx] = event['strain']
            physics_ds[idx] = event['physics_features']
            xyz_ds[idx] = event['xyz']
            ra_ds[idx] = event['ra']
            dec_ds[idx] = event['dec']
            healpix_ds[idx] = healpix_label

    print(f"\n✅ Preprocessing complete!")
    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024**3):.2f} GB")

    # Verify
    print("\nVerifying saved data...")
    with h5py.File(output_file, 'r') as f:
        print(f"  Events: {f['strain'].shape[0]}")
        print(f"  Strain shape: {f['strain'].shape}")
        print(f"  Physics features shape: {f['physics_features'].shape}")
        print(f"  XYZ shape: {f['xyz'].shape}")

        # Check for NaN/Inf
        has_nan = np.any(np.isnan(f['strain'][:100]))
        has_inf = np.any(np.isinf(f['strain'][:100]))
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")

        if has_nan or has_inf:
            print("  ⚠️ WARNING: Data contains NaN or Inf values!")
        else:
            print("  ✅ Data looks good!")

    print("\n" + "="*80)
    print("Next steps:")
    print(f"1. Update your training config to use: {output_dir}")
    print("2. Training will load preprocessed data (instant!)")
    print("3. No more bandpass filtering during training")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess GW dataset and save to HDF5'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory with .npy files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/data/stu_bonhomme/tfe/dataset/romain_preprocessed',
        help='Output directory for HDF5 file'
    )
    parser.add_argument(
        '--target-length',
        type=int,
        default=2048,
        help='Target sequence length'
    )
    parser.add_argument(
        '--bandpass-low',
        type=float,
        default=30.0,
        help='Low frequency cutoff (Hz)'
    )
    parser.add_argument(
        '--bandpass-high',
        type=float,
        default=1024.0,
        help='High frequency cutoff (Hz)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable normalization'
    )
    parser.add_argument(
        '--healpix-nside',
        type=int,
        default=8,
        help='HEALPix nside parameter'
    )

    args = parser.parse_args()

    preprocess_and_save(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_length=args.target_length,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        normalize=not args.no_normalize,
        healpix_nside=args.healpix_nside,
    )


if __name__ == '__main__':
    main()
