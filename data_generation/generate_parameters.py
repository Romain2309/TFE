#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
from uuid import uuid4


def generate_advanced_params(n_events, output_file):
    np.random.seed(42)
    params = []
    for event_id in range(n_events):
        for detector in ['H1', 'L1', 'V1']:
            ra = np.random.uniform(0, 2 * np.pi)
            sin_dec = np.random.uniform(-1, 1)
            dec = np.arcsin(sin_dec)
            params.append({
                'event_id': event_id,
                'detector': detector,
                'seed': event_id * 3 + {'H1': 0, 'L1': 1, 'V1': 2}[detector],
                'duration': np.random.randint(50, 1001),
                'chirp_duration': np.random.randint(50, 501),
                'f0': np.random.uniform(50, 700),
                'f1': np.random.uniform(500, 1200),
                'f_ev': np.random.choice(['linear', 'logarithmic', 'quadratic', 'hyperbolic']),
                'hrss': np.random.uniform(1e-19, 1e-18),
                'delay': np.random.randint(10, 100),
                'ra': ra,
                'dec': dec
            })
    df = pd.DataFrame(params)
    df.to_csv(output_file, index=False)


def generate_benchmark_small_params(n_events, output_file):
    np.random.seed(42)
    fixed_configs = [
        {'duration': 100, 'chirp_duration': 20, 'f0': 50.0, 'f1': 500.0, 'f_ev': 'linear', 'hrss': 1e-19, 'delay': 40},
        {'duration': 100, 'chirp_duration': 40, 'f0': 300.0, 'f1': 750.0, 'f_ev': 'quadratic', 'hrss': 2.5e-19, 'delay': 30},
        {'duration': 100, 'chirp_duration': 60, 'f0': 500.0, 'f1': 1000.0, 'f_ev': 'logarithmic', 'hrss': 5e-19, 'delay': 20},
        {'duration': 100, 'chirp_duration': 80, 'f0': 700.0, 'f1': 1200.0, 'f_ev': 'hyperbolic', 'hrss': 7.5e-19, 'delay': 10}
    ]
    params = []
    event_id = 0
    for config_id, config in enumerate(fixed_configs):
        for _ in range(n_events // 4):
            ra = np.random.uniform(0, 2 * np.pi)
            sin_dec = np.random.uniform(-1, 1)
            dec = np.arcsin(sin_dec)
            params.append({
                'event_id': event_id,
                'config_id': config_id,
                'noise_seed': 42,
                **config,
                'ra': ra,
                'dec': dec,
                'uuid': str(uuid4())[:8]
            })
            event_id += 1
    df = pd.DataFrame(params)
    df.to_csv(output_file, index=False)


def generate_benchmark_same_small_params(n_events, output_file):
    np.random.seed(42)
    fixed_config = {
        'config_id': 1,
        'duration': 250,
        'chirp_duration': 100,
        'f0': 30.0,
        'f1': 1000.0,
        'f_ev': 'logarithmic',
        'hrss': 1e-18,
        'delay': 50
    }
    params = []
    for event_id in range(n_events):
        ra = np.random.uniform(0, 2 * np.pi)
        sin_dec = np.random.uniform(-1, 1)
        dec = np.arcsin(sin_dec)
        params.append({
            'event_id': event_id,
            **fixed_config,
            'ra': ra,
            'dec': dec,
            'uuid': str(uuid4())[:8]
        })
    df = pd.DataFrame(params)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['advanced', 'benchmark_small', 'benchmark_same_small'])
    parser.add_argument('--n_events', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    if args.dataset == 'advanced':
        generate_advanced_params(args.n_events, args.output)
    elif args.dataset == 'benchmark_small':
        generate_benchmark_small_params(args.n_events, args.output)
    elif args.dataset == 'benchmark_same_small':
        generate_benchmark_same_small_params(args.n_events, args.output)
