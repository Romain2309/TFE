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
from gw_skylocalization.evaluation.metrics import compute_angular_errors
from gw_skylocalization.utils.coordinates import xyz_to_radec

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False
    print("Warning: healpy not installed, sky map plots will be skipped")


def load_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config.get('model_type', 'regressor')
        n_pixels = config.get('n_pixels', 768)
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'head.classifier.0.weight' in state_dict:
            model_type = 'classifier'
            n_pixels = state_dict['head.classifier.3.bias'].shape[0]
        elif 'head.prob_net.0.weight' in state_dict:
            model_type = 'probmap'
            n_pixels = state_dict['head.prob_net.5.bias'].shape[0]
        elif 'head.tau_head.0.weight' in state_dict:
            model_type = 'multitask'
            n_pixels = 768
        else:
            model_type = 'regressor'
            n_pixels = 768
    print(f"Loading {model_type} model (n_pixels={n_pixels})...")

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

def evaluate_regressor(model, test_loader, device='cuda', max_samples=None):
    all_predictions = []
    all_targets = []
    all_errors = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            strain = batch['strain'].to(device)
            physics_features = batch['physics_features'].to(device)
            xyz_target = batch['xyz'].to(device)

            xyz_pred = model(strain, physics_features)

            errors = compute_angular_errors(xyz_pred.cpu().numpy(), xyz_target.cpu().numpy(), degrees=True)

            all_predictions.append(xyz_pred.cpu().numpy())
            all_targets.append(xyz_target.cpu().numpy())
            all_errors.append(errors)

            n_samples += len(errors)
            if max_samples and n_samples >= max_samples:
                break

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    angular_errors = np.concatenate(all_errors, axis=0)

    return {
        'predictions': predictions,
        'targets': targets,
        'angular_errors': angular_errors,
    }

def evaluate_classifier(model, test_loader, device='cuda', max_samples=None, nside=8):
    all_predictions = []
    all_targets = []
    all_errors = []
    all_predicted_pixels = []
    all_target_pixels = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            strain = batch['strain'].to(device)
            physics_features = batch['physics_features'].to(device)
            xyz_target = batch['xyz'].to(device)
            healpix_target = batch['healpix_label'].to(device)

            logits = model(strain, physics_features)
            predicted_pixels = torch.argmax(logits, dim=1)

            predicted_pixels_np = predicted_pixels.cpu().numpy()
            xyz_pred = np.zeros((len(predicted_pixels_np), 3))
            for i, pix in enumerate(predicted_pixels_np):
                theta, phi = hp.pix2ang(nside, int(pix))
                xyz_pred[i, 0] = np.sin(theta) * np.cos(phi)
                xyz_pred[i, 1] = np.sin(theta) * np.sin(phi)
                xyz_pred[i, 2] = np.cos(theta)

            errors = compute_angular_errors(xyz_pred, xyz_target.cpu().numpy(), degrees=True)

            all_predictions.append(xyz_pred)
            all_targets.append(xyz_target.cpu().numpy())
            all_errors.append(errors)
            all_predicted_pixels.append(predicted_pixels_np)
            all_target_pixels.append(healpix_target.cpu().numpy())

            n_samples += len(errors)
            if max_samples and n_samples >= max_samples:
                break

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    angular_errors = np.concatenate(all_errors, axis=0)
    predicted_pixels = np.concatenate(all_predicted_pixels, axis=0)
    target_pixels = np.concatenate(all_target_pixels, axis=0)

    accuracy = (predicted_pixels == target_pixels).mean()

    return {
        'predictions': predictions,
        'targets': targets,
        'angular_errors': angular_errors,
        'predicted_pixels': predicted_pixels,
        'target_pixels': target_pixels,
        'accuracy': accuracy,
    }

def evaluate_multitask(model, test_loader, device='cuda', max_samples=None):
    all_predictions = []
    all_targets = []
    all_errors = []
    all_tau_pred = []
    all_tau_target = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            strain = batch['strain'].to(device)
            physics_features = batch['physics_features'].to(device)
            xyz_target = batch['xyz'].to(device)

            tau_target = physics_features[:, :2]

            xyz_pred, tau_pred = model(strain, physics_features)

            errors = compute_angular_errors(xyz_pred.cpu().numpy(), xyz_target.cpu().numpy(), degrees=True)

            all_predictions.append(xyz_pred.cpu().numpy())
            all_targets.append(xyz_target.cpu().numpy())
            all_errors.append(errors)
            all_tau_pred.append(tau_pred.cpu().numpy())
            all_tau_target.append(tau_target.cpu().numpy())

            n_samples += len(errors)
            if max_samples and n_samples >= max_samples:
                break

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    angular_errors = np.concatenate(all_errors, axis=0)
    tau_pred = np.concatenate(all_tau_pred, axis=0)
    tau_target = np.concatenate(all_tau_target, axis=0)

    tau_errors = np.abs(tau_pred - tau_target)

    return {
        'predictions': predictions,
        'targets': targets,
        'angular_errors': angular_errors,
        'tau_pred': tau_pred,
        'tau_target': tau_target,
        'tau_errors': tau_errors,
    }

def plot_angular_error_histogram(angular_errors, output_path, title='Angular Error Distribution'):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(angular_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    median = np.median(angular_errors)
    mean = np.mean(angular_errors)
    p90 = np.percentile(angular_errors, 90)

    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}°')
    ax.axvline(mean, color='green', linestyle=':', linewidth=2, label=f'Mean: {mean:.2f}°')
    ax.axvline(p90, color='orange', linestyle='-.', linewidth=2, label=f'90th percentile: {p90:.2f}°')

    ax.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path.name}")

def plot_sky_comparison(predictions, targets, angular_errors, output_path, n_samples=200):
    pred_ra, pred_dec = xyz_to_radec(predictions[:n_samples])
    true_ra, true_dec = xyz_to_radec(targets[:n_samples])
    errors = angular_errors[:n_samples]

    pred_ra_deg = np.degrees(pred_ra)
    pred_dec_deg = np.degrees(pred_dec)
    true_ra_deg = np.degrees(true_ra)
    true_dec_deg = np.degrees(true_dec)

    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'projection': 'mollweide'})

    pred_ra_rad = np.radians(pred_ra_deg) - np.pi
    true_ra_rad = np.radians(true_ra_deg) - np.pi
    pred_dec_rad = np.radians(pred_dec_deg)
    true_dec_rad = np.radians(true_dec_deg)

    ax.scatter(true_ra_rad, true_dec_rad, c='blue', s=30, alpha=0.6, label='True', zorder=2)

    sc = ax.scatter(pred_ra_rad, pred_dec_rad, c=errors, cmap='Reds', s=30, alpha=0.6, vmin=0, vmax=np.percentile(errors, 95), zorder=3)

    for i in range(min(100, n_samples)):
        ax.plot([true_ra_rad[i], pred_ra_rad[i]], [true_dec_rad[i], pred_dec_rad[i]], 'k-', alpha=0.15, linewidth=0.5, zorder=1)

    cbar = plt.colorbar(sc, label='Angular Error (°)', shrink=0.6, pad=0.05)
    ax.set_title('Sky Localization: True (blue) vs Predicted (colored by error)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path.name}")

def plot_error_vs_position(predictions, targets, angular_errors, output_path):
    true_ra, true_dec = xyz_to_radec(targets)
    true_ra_deg = np.degrees(true_ra)
    true_dec_deg = np.degrees(true_dec)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(true_ra_deg, angular_errors, alpha=0.3, s=10, c='steelblue')
    axes[0].set_xlabel('True Right Ascension (degrees)', fontsize=12)
    axes[0].set_ylabel('Angular Error (degrees)', fontsize=12)
    axes[0].set_title('Error vs Right Ascension', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(np.median(angular_errors), color='red', linestyle='--',
                    alpha=0.7, label=f'Median: {np.median(angular_errors):.2f}°')
    axes[0].legend()

    axes[1].scatter(true_dec_deg, angular_errors, alpha=0.3, s=10, c='steelblue')
    axes[1].set_xlabel('True Declination (degrees)', fontsize=12)
    axes[1].set_ylabel('Angular Error (degrees)', fontsize=12)
    axes[1].set_title('Error vs Declination', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(np.median(angular_errors), color='red', linestyle='--',
                    alpha=0.7, label=f'Median: {np.median(angular_errors):.2f}°')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path.name}")

def plot_cumulative_distribution(angular_errors, output_path):
    sorted_errors = np.sort(angular_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(sorted_errors, cumulative * 100, 'b-', linewidth=2)
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90th percentile')

    p50 = np.percentile(angular_errors, 50)
    p90 = np.percentile(angular_errors, 90)
    ax.axvline(p50, color='red', linestyle=':', alpha=0.5)
    ax.axvline(p90, color='orange', linestyle=':', alpha=0.5)

    ax.text(p50, 5, f'{p50:.1f}°', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(p90, 5, f'{p90:.1f}°', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Angular Errors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(100, np.percentile(angular_errors, 99)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path.name}")

def print_statistics(results, output_path):
    errors = results['angular_errors']

    stats_text = []
    stats_text.append("="*80)
    stats_text.append("EVALUATION STATISTICS")
    stats_text.append("="*80)
    stats_text.append(f"\nNumber of samples: {len(errors)}")
    stats_text.append(f"\nAngular Error Statistics:")
    stats_text.append(f"  Median:  {np.median(errors):.3f}°")
    stats_text.append(f"  Mean:    {np.mean(errors):.3f}°")
    stats_text.append(f"  Std:     {np.std(errors):.3f}°")
    stats_text.append(f"  Min:     {np.min(errors):.3f}°")
    stats_text.append(f"  Max:     {np.max(errors):.3f}°")
    stats_text.append(f"\nPercentiles:")
    stats_text.append(f"  25th:    {np.percentile(errors, 25):.3f}°")
    stats_text.append(f"  50th:    {np.percentile(errors, 50):.3f}°")
    stats_text.append(f"  75th:    {np.percentile(errors, 75):.3f}°")
    stats_text.append(f"  90th:    {np.percentile(errors, 90):.3f}°")
    stats_text.append(f"  95th:    {np.percentile(errors, 95):.3f}°")
    stats_text.append(f"  99th:    {np.percentile(errors, 99):.3f}°")
    stats_text.append(f"\nPerformance:")
    stats_text.append(f"  < 5°:    {(errors < 5).sum() / len(errors) * 100:.1f}%")
    stats_text.append(f"  < 10°:   {(errors < 10).sum() / len(errors) * 100:.1f}%")
    stats_text.append(f"  < 20°:   {(errors < 20).sum() / len(errors) * 100:.1f}%")
    stats_text.append("="*80)

    with open(output_path, 'w') as f:
        f.write('\n'.join(stats_text))

def main():
    parser = argparse.ArgumentParser(description='Evaluate Romain model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--hdf5-path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples to evaluate (None=all)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, model_type, n_pixels = load_model(args.model_path, device)

    _, _, test_loader = create_hdf5_dataloaders(hdf5_path=args.hdf5_path, batch_size=args.batch_size, train_frac=0.7, val_frac=0.15, test_frac=0.15, num_workers=4, verbose=True)

    nside = int((n_pixels / 12) ** 0.5) if model_type in ['classifier', 'probmap'] else 8

    if model_type == 'regressor':
        results = evaluate_regressor(model, test_loader, device, max_samples=args.n_samples)
    elif model_type == 'classifier':
        results = evaluate_classifier(model, test_loader, device, max_samples=args.n_samples, nside=nside)
    elif model_type == 'multitask':
        results = evaluate_multitask(model, test_loader, device, max_samples=args.n_samples)
    elif model_type == 'probmap':
        print("ProbabilityMap models should be evaluated with evaluate_with_skymaps.py")
        print("This script only handles regressor, classifier, and multitask models.")
        sys.exit(1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    save_dict = {
        'predictions': results['predictions'],
        'targets': results['targets'],
        'angular_errors': results['angular_errors'],
    }

    if model_type == 'classifier':
        save_dict['predicted_pixels'] = results['predicted_pixels']
        save_dict['target_pixels'] = results['target_pixels']
        save_dict['accuracy'] = results['accuracy']
    elif model_type == 'multitask':
        save_dict['tau_pred'] = results['tau_pred']
        save_dict['tau_target'] = results['tau_target']
        save_dict['tau_errors'] = results['tau_errors']

    np.savez(output_dir / 'predictions.npz', **save_dict)

    print_statistics(results, output_dir / 'statistics.txt')
    plot_angular_error_histogram(results['angular_errors'], output_dir / 'error_histogram.pdf')
    plot_sky_comparison(results['predictions'], results['targets'], results['angular_errors'], output_dir / 'sky_comparison.pdf', n_samples=200)
    plot_error_vs_position(results['predictions'], results['targets'], results['angular_errors'], output_dir / 'error_vs_position.pdf')
    plot_cumulative_distribution(results['angular_errors'], output_dir / 'cumulative_distribution.pdf')

if __name__ == '__main__':
    main()
