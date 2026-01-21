#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gw_skylocalization.data.file_readers import PreGeneratedDataReader
from gw_skylocalization.utils.coordinates import radec_to_healpix

def preprocess_and_save(input_dir, output_dir, target_length = 2048, bandpass_low = 30.0, bandpass_high = 1024.0, normalize = True, healpix_nside = 8):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reader = PreGeneratedDataReader(data_dir=input_dir, target_length=target_length, bandpass=(bandpass_low, bandpass_high), normalize=normalize, verbose=True)
    output_file = output_path / 'preprocessed_data.h5'

    import sys
    sys.stdout.flush()

    with h5py.File(output_file, 'w') as f:
        max_chunk_bytes = 3.5 * 1024**3
        bytes_per_event = 3 * target_length * 4
        chunk_events = min(100, int(max_chunk_bytes / bytes_per_event))
        chunk_events = max(1, chunk_events)

        strain_ds = f.create_dataset('strain', shape=(n_events, 3, target_length), dtype=np.float32, chunks=(chunk_events, 3, target_length))

        physics_ds = f.create_dataset('physics_features', shape=(n_events, 6), dtype=np.float32)

        xyz_ds = f.create_dataset('xyz', shape=(n_events, 3), dtype=np.float32)

        ra_ds = f.create_dataset('ra', shape=(n_events,), dtype=np.float32)

        dec_ds = f.create_dataset('dec', shape=(n_events,), dtype=np.float32)

        healpix_ds = f.create_dataset('healpix_label', shape=(n_events,), dtype=np.int32)

        f.attrs['n_events'] = n_events
        f.attrs['target_length'] = target_length
        f.attrs['bandpass_low'] = bandpass_low
        f.attrs['bandpass_high'] = bandpass_high
        f.attrs['normalize'] = normalize
        f.attrs['healpix_nside'] = healpix_nside

        for idx in tqdm(range(n_events), desc="Preprocessing"):
            event = reader.load_event(idx)

            healpix_label = radec_to_healpix(event['ra'], event['dec'], nside=healpix_nside)

            strain_ds[idx] = event['strain']
            physics_ds[idx] = event['physics_features']
            xyz_ds[idx] = event['xyz']
            ra_ds[idx] = event['ra']
            dec_ds[idx] = event['dec']
            healpix_ds[idx] = healpix_label

    with h5py.File(output_file, 'r') as f:
        print(f"  Events: {f['strain'].shape[0]}")
        print(f"  Strain shape: {f['strain'].shape}")
        print(f"  Physics features shape: {f['physics_features'].shape}")
        print(f"  XYZ shape: {f['xyz'].shape}")

        has_nan = np.any(np.isnan(f['strain'][:100]))
        has_inf = np.any(np.isinf(f['strain'][:100]))
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess GW dataset and save to HDF5')

    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with .npy files')
    parser.add_argument('--output-dir', type=str, default='/data/stu_bonhomme/tfe/dataset/romain_preprocessed', help='Output directory for HDF5 file')
    parser.add_argument('--target-length', type=int,default=2048, help='Target sequence length')
    parser.add_argument('--bandpass-low', type=float, default=30.0, help='Low frequency cutoff (Hz)')
    parser.add_argument('--bandpass-high', type=float, default=1024.0, help='High frequency cutoff (Hz)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization')
    parser.add_argument('--healpix-nside', type=int, default=8, help='HEALPix nside parameter')

    args = parser.parse_args()

    preprocess_and_save(input_dir=args.input_dir, output_dir=args.output_dir, target_length=args.target_length, bandpass_low=args.bandpass_low,
                        bandpass_high=args.bandpass_high, normalize=not args.no_normalize, healpix_nside=args.healpix_nside,)


if __name__ == '__main__':
    main()
