#!/usr/bin/env python3
import argparse
import subprocess
import pandas as pd
import os


def generate_dataset(dataset_type, n_events, output_dir, n_jobs=1):
    os.makedirs(output_dir, exist_ok=True)
    param_file = os.path.join(output_dir, 'parameters.csv')
    subprocess.run([
        'python', 'generate_parameters.py',
        '--dataset', dataset_type,
        '--n_events', str(n_events),
        '--output', param_file
    ], check=True)
    params_df = pd.read_csv(param_file)
    for idx, row in params_df.iterrows():
        cmd = ['python', 'simulate_event.py', '--dataset', dataset_type]
        for col, val in row.items():
            if pd.notna(val):
                cmd.extend([f'--{col}', str(val)])
        cmd.extend(['--output_dir', output_dir])
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['advanced', 'benchmark_small', 'benchmark_same_small'])
    parser.add_argument('--n_events', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    generate_dataset(args.dataset, args.n_events, args.output_dir, args.n_jobs)
