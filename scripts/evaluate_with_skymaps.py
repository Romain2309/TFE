#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from gw_skylocalization.models.networks import SkyRegressor, SkyClassifier, SkyMultiTask, SkyProbabilityMap
from gw_skylocalization.data.hdf5_loader import create_hdf5_dataloaders
from gw_skylocalization.evaluation.metrics import compute_angular_errors, searched_area
from gw_skylocalization.utils.coordinates import xyz_to_radec, radec_to_healpix

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False
    sys.exit(1)

def load_model(model_path, device='cuda'):

    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config.get('model_type', 'regressor')
        n_pixels = config.get('n_pixels', 768)
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'head.prob_net.0.weight' in state_dict:
            model_type = 'probmap'
            n_pixels = state_dict['head.prob_net.5.bias'].shape[0]
        elif 'head.classifier.0.weight' in state_dict:
            model_type = 'classifier'
            n_pixels = state_dict['head.classifier.3.bias'].shape[0]
        elif 'head.tau_head.0.weight' in state_dict:
            model_type = 'multitask'
            n_pixels = 768
        else:
            model_type = 'regressor'
            n_pixels = 768

    if model_type == 'regressor':
        model = SkyRegressor()
    elif model_type == 'classifier':
        model = SkyClassifier(n_pixels=n_pixels)
    elif model_type == 'multitask':
        model = SkyMultiTask()
    elif model_type == 'probmap':
        model = SkyProbabilityMap(n_pixels=n_pixels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, model_type, n_pixels

def evaluate_probmap(model, test_loader, device='cuda', max_samples=None, nside=8):

    all_probs = []
    all_true_pixels = []
    all_true_xyz = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            strain = batch['strain'].to(device)
            physics_features = batch['physics_features'].to(device)
            xyz_target = batch['xyz']
            healpix_target = batch['healpix_label']

            probs = model(strain, physics_features, return_probs=True)

            all_probs.append(probs.cpu().numpy())
            all_true_pixels.append(healpix_target.cpu().numpy())
            all_true_xyz.append(xyz_target.cpu().numpy())

            n_samples += len(probs)
            if max_samples and n_samples >= max_samples:
                break

    probs = np.concatenate(all_probs, axis=0)
    true_pixels = np.concatenate(all_true_pixels, axis=0)
    true_xyz = np.concatenate(all_true_xyz, axis=0)

    return {
        'probs': probs,
        'true_pixels': true_pixels,
        'true_xyz': true_xyz,
        'nside': nside,
    }

def evaluate_regressor(model, test_loader, device='cuda', max_samples=None):

    all_predictions = []
    all_targets = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            strain = batch['strain'].to(device)
            physics_features = batch['physics_features'].to(device)
            xyz_target = batch['xyz']

            xyz_pred = model(strain, physics_features)

            all_predictions.append(xyz_pred.cpu().numpy())
            all_targets.append(xyz_target.cpu().numpy())

            n_samples += len(xyz_pred)
            if max_samples and n_samples >= max_samples:
                break

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    angular_errors = compute_angular_errors(predictions, targets, degrees=True)

    return {
        'predictions': predictions,
        'targets': targets,
        'angular_errors': angular_errors,
    }

def plot_probability_skymap(probs, nside, true_ra, true_dec, output_path, title=None,
                            show_contours=True, smoothing_fwhm_deg=2.0):

    if smoothing_fwhm_deg > 0:
        probs_plot = hp.smoothing(probs, fwhm=np.radians(smoothing_fwhm_deg))
        probs_plot = np.maximum(probs_plot, 0)
        if probs_plot.sum() > 0:
            probs_plot = probs_plot / probs_plot.sum()
    else:
        probs_plot = probs.copy()

    pred_pixel = probs.argmax()
    pred_theta, pred_phi = hp.pix2ang(nside, pred_pixel)
    pred_ra = np.degrees(pred_phi)
    pred_dec = 90 - np.degrees(pred_theta)

    true_theta = np.radians(90 - true_dec)
    true_phi = np.radians(true_ra)
    angular_dist_rad = hp.rotator.angdist([true_theta, true_phi],
                                          [pred_theta, pred_phi])
    if isinstance(angular_dist_rad, np.ndarray):
        angular_dist_rad = float(angular_dist_rad.item())
    angular_dist = np.degrees(angular_dist_rad)

    fig = plt.figure(figsize=(14, 8))

    hp.mollview(
        probs_plot,
        title='',
        cmap='hot',
        hold=True,
        fig=fig.number,
        xsize=2000,
        min=0,
        max=probs_plot.max() * 0.9,
        unit='Probability',
    )

    if show_contours and probs_plot.sum() > 0:
        sorted_idx = np.argsort(probs_plot)[::-1]
        cumsum = np.cumsum(probs_plot[sorted_idx])

        for level, color, ls, lw in [(0.5, 'cyan', '-', 2.5), (0.9, 'lime', '--', 2.0)]:
            n_pixels = np.searchsorted(cumsum, level) + 1
            n_pixels = min(n_pixels, len(sorted_idx) - 1)
            threshold = probs_plot[sorted_idx[n_pixels]]

            contour_map = np.where(probs_plot >= threshold, 1.0, 0.0)
            try:
                hp.projcontour(contour_map, levels=[0.5], colors=[color],
                              linestyles=[ls], linewidths=[lw],
                              label=f'{int(level*100)}% credible region')
            except Exception:
                pass

    hp.projscatter(
        true_ra, true_dec,
        lonlat=True,
        marker='*',
        c='white',
        s=400,
        edgecolors='black',
        linewidths=2,
        zorder=10,
        label='True position'
    )

    hp.projscatter(
        pred_ra, pred_dec,
        lonlat=True,
        marker='+',
        c='red',
        s=300,
        linewidths=3,
        zorder=10,
        label='Max probability'
    )

    hp.graticule(dpar=30, dmer=30, alpha=0.4, color='gray')

    if title:
        plt.title(f'{title}\nAngular Error: {angular_dist:.2f}°', fontsize=14, pad=20)
    else:
        plt.title(f'Angular Error: {angular_dist:.2f}°', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_skymap_with_zoom(probs, nside, true_ra, true_dec, output_path):

    probs_plot = hp.smoothing(probs, fwhm=np.radians(2.0))
    probs_plot = np.maximum(probs_plot, 0)
    if probs_plot.sum() > 0:
        probs_plot = probs_plot / probs_plot.sum()

    pred_pixel = probs.argmax()
    pred_theta, pred_phi = hp.pix2ang(nside, pred_pixel)
    pred_ra = np.degrees(pred_phi)
    pred_dec = 90 - np.degrees(pred_theta)

    fig = plt.figure(figsize=(18, 7))

    hp.mollview(
        probs_plot,
        title='Full Sky Probability Map',
        cmap='hot',
        hold=True,
        sub=(1, 2, 1),
        xsize=1500,
        min=0,
        max=probs_plot.max() * 0.9,
    )
    hp.projscatter(true_ra, true_dec, lonlat=True, marker='*',
                   c='white', s=300, edgecolors='black', linewidths=1.5)
    hp.graticule(dpar=30, dmer=30, alpha=0.3)

    hp.gnomview(
        probs_plot,
        rot=(true_ra, true_dec, 0),
        reso=1.5,
        xsize=600,
        title='Zoomed View (centered on true position)',
        cmap='hot',
        hold=True,
        sub=(1, 2, 2),
        min=0,
        max=probs_plot.max() * 0.9,
    )
    hp.projscatter(true_ra, true_dec, lonlat=True, marker='*',
                   c='white', s=400, edgecolors='black', linewidths=2)
    hp.projscatter(pred_ra, pred_dec, lonlat=True, marker='+',
                   c='red', s=300, linewidths=3)
    hp.graticule(dpar=10, dmer=10, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def compute_skymap_metrics(probs, true_pixels, nside):

    n_samples = len(probs)

    searched_areas = []
    credible_areas_50 = []
    credible_areas_90 = []
    angular_errors = []

    for i in range(n_samples):
        metrics = searched_area(probs[i], true_pixels[i], nside, credible_levels=(0.5, 0.9))
        searched_areas.append(metrics['searched_area_deg2'])
        credible_areas_50.append(metrics['area_50_deg2'])
        credible_areas_90.append(metrics['area_90_deg2'])

        pred_pixel = probs[i].argmax()
        pred_theta, pred_phi = hp.pix2ang(nside, pred_pixel)
        true_theta, true_phi = hp.pix2ang(nside, true_pixels[i])

        angular_dist = np.degrees(hp.rotator.angdist([true_theta, true_phi],
                                                      [pred_theta, pred_phi]))
        angular_errors.append(angular_dist)

    return {
        'searched_areas': np.array(searched_areas),
        'credible_area_50': np.array(credible_areas_50),
        'credible_area_90': np.array(credible_areas_90),
        'angular_errors': np.array(angular_errors),
    }

def plot_searched_area_histogram(searched_areas, output_path, credible_level=90):

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(searched_areas, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    median = np.median(searched_areas)
    mean = np.mean(searched_areas)

    ax.axvline(median, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median:.1f} deg²')
    ax.axvline(mean, color='green', linestyle=':', linewidth=2,
               label=f'Mean: {mean:.1f} deg²')

    ax.set_xlabel('Searched Area (deg²)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{credible_level}% Credible Region Searched Area Distribution',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate with sky maps')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--hdf5-path', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_skymaps',
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of test samples to evaluate')
    parser.add_argument('--n-skymap-samples', type=int, default=10,
                        help='Number of individual sky maps to generate')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, model_type, n_pixels = load_model(args.model_path, device)

    _, _, test_loader = create_hdf5_dataloaders(
        hdf5_path=args.hdf5_path,
        batch_size=args.batch_size,
        verbose=True,
    )

    nside = int((n_pixels / 12) ** 0.5) if model_type in ['probmap', 'classifier'] else 8

    if model_type == 'probmap':

        results = evaluate_probmap(model, test_loader, device, args.n_samples, nside=nside)

        metrics = compute_skymap_metrics(results['probs'], results['true_pixels'], results['nside'])

        np.savez(
            output_dir / 'skymap_metrics.npz',
            searched_areas=metrics['searched_areas'],
            credible_area_50=metrics['credible_area_50'],
            credible_area_90=metrics['credible_area_90'],
            angular_errors=metrics['angular_errors'],
        )

        plot_searched_area_histogram(
            metrics['searched_areas'],
            output_dir / 'searched_area_histogram.pdf',
            credible_level='actual'
        )
        plot_searched_area_histogram(
            metrics['credible_area_50'],
            output_dir / 'credible_area_50_histogram.pdf',
            credible_level=50
        )
        plot_searched_area_histogram(
            metrics['credible_area_90'],
            output_dir / 'credible_area_90_histogram.pdf',
            credible_level=90
        )

        n_show = min(args.n_skymap_samples, len(results['probs']))

        for i in tqdm(range(n_show), desc="Sky maps"):
            true_theta, true_phi = hp.pix2ang(results['nside'], results['true_pixels'][i])
            true_ra = np.degrees(true_phi)
            true_dec = 90 - np.degrees(true_theta)

            plot_probability_skymap(
                results['probs'][i],
                results['nside'],
                true_ra, true_dec,
                output_dir / f'skymap_{i:03d}.pdf',
                title=f'Sample {i}',
            )

            plot_skymap_with_zoom(
                results['probs'][i],
                results['nside'],
                true_ra, true_dec,
                output_dir / f'skymap_zoom_{i:03d}.pdf',
            )

    else:

        results = evaluate_regressor(model, test_loader, device, args.n_samples)

if __name__ == '__main__':
    main()
