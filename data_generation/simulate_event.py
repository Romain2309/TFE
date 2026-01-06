#!/usr/bin/env python3
import numpy as np
import os
import argparse
import copy
from GWpyxel.pyxel import Pyxel
from gwpy.timeseries import TimeSeries


def simulate_advanced(args):
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    pyxel = Pyxel()
    noise_psd = 'aLIGOZeroDetHighPower'
    pyxel.fetch_data(args.detector, noise_psd, duration=args.duration, cache=False)
    waveform = f'chirp~{args.chirp_duration}~{args.f0:.2f}~{args.f1:.2f}~{args.f_ev}~0.4'
    pyxel.inject_waveform(waveform, delay=args.delay, hrss=args.hrss, right_ascension=args.ra, declination=args.dec)
    filename = (f"EVENT_ID_{args.event_id}_{args.detector}_seed{args.seed}_"
                f"noiseDur{args.duration}s_chirpDur{args.chirp_duration}s_"
                f"f0-{args.f0:.2f}_f1-{args.f1:.2f}_{args.f_ev}_"
                f"hrss-{args.hrss:.2e}_delay-{args.delay}_"
                f"RA-{args.ra:.3f}_DEC-{args.dec:.3f}.npy")
    np.save(os.path.join(args.output_dir, filename), pyxel.data, allow_pickle=True)


def simulate_benchmark_small(args):
    np.random.seed(args.noise_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    pyxel = Pyxel()
    detectors = ['H1', 'L1', 'V1']
    for detector in detectors:
        pyxel.fetch_data(detector, 'aLIGOZeroDetHighPower', duration=args.duration, cache=False)
    waveform = f'chirp~{args.chirp_duration}~{args.f0:.2f}~{args.f1:.2f}~{args.f_ev}'
    pyxel.inject_waveform(waveform, delay=args.delay, hrss=args.hrss, right_ascension=args.ra, declination=args.dec, overwrite='False')
    for detector in detectors:
        pyxel_copy = copy.deepcopy(pyxel.data)
        for ifo in detectors:
            if ifo != detector:
                del pyxel_copy[ifo]
        filename = (f"EVENT_{args.event_id:05d}_CONFIG_{args.config_id}_{detector}_"
                   f"RA_{args.ra:.3f}_DEC_{args.dec:.3f}_{args.uuid}.npy")
        np.save(os.path.join(args.output_dir, filename), pyxel_copy, allow_pickle=True)


def simulate_benchmark_same_small(args):
    os.makedirs(args.output_dir, exist_ok=True)
    pyxel = Pyxel()
    detectors = ['H1', 'L1', 'V1']
    np.random.seed(args.event_id + 123456)
    for detector in detectors:
        pyxel.fetch_data(detector, 'aLIGOZeroDetHighPower', duration=args.duration, cache=False)
        ts_ligo = pyxel.data[detector]
        sample_rate = ts_ligo.sample_rate.value
        n_samples = len(ts_ligo)
        t0 = ts_ligo.t0.value
        gaussian_noise_std = 5e-22
        gaussian_noise = np.random.randn(n_samples) * gaussian_noise_std
        ts_gaussian = TimeSeries(gaussian_noise, sample_rate=sample_rate, t0=t0, name=detector, unit='strain')
        pyxel.data[detector] = ts_gaussian
    waveform = f'chirp~{args.chirp_duration}~{args.f0:.2f}~{args.f1:.2f}~{args.f_ev}'
    pyxel.inject_waveform(waveform, delay=args.delay, hrss=args.hrss, right_ascension=args.ra, declination=args.dec, overwrite='False')
    for detector in detectors:
        pyxel_copy = copy.deepcopy(pyxel.data)
        for ifo in detectors:
            if ifo != detector:
                del pyxel_copy[ifo]
        filename = (f"EVENT_{args.event_id:05d}_CONFIG_{args.config_id}_{detector}_"
                   f"RA_{args.ra:.3f}_DEC_{args.dec:.3f}_{args.uuid}.npy")
        np.save(os.path.join(args.output_dir, filename), pyxel_copy, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['advanced', 'benchmark_small', 'benchmark_same_small'])
    parser.add_argument('--event_id', type=int, required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--chirp_duration', type=int, required=True)
    parser.add_argument('--f0', type=float, required=True)
    parser.add_argument('--f1', type=float, required=True)
    parser.add_argument('--f_ev', type=str, required=True)
    parser.add_argument('--hrss', type=float, required=True)
    parser.add_argument('--delay', type=int, required=True)
    parser.add_argument('--ra', type=float, required=True)
    parser.add_argument('--dec', type=float, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--detector', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--config_id', type=int)
    parser.add_argument('--noise_seed', type=int)
    parser.add_argument('--uuid', type=str)
    args = parser.parse_args()
    if args.dataset == 'advanced':
        simulate_advanced(args)
    elif args.dataset == 'benchmark_small':
        simulate_benchmark_small(args)
    elif args.dataset == 'benchmark_same_small':
        simulate_benchmark_same_small(args)
