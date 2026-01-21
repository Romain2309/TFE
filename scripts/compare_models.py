#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

def load_results(results_dir):
    results = {}

    model_types = ['regressor', 'classifier', 'multitask', 'probmap']

    for model_type in model_types:
        if model_type == 'regressor':
            patterns = [f"{model_type}/{model_type}_*", f"{model_type}_nside16/{model_type}_*", f"{model_type}_training/{model_type}_*", f"{model_type}_full_training/{model_type}_*"]
        elif model_type == 'probmap':
            patterns = [f"{model_type}/{model_type}_*", f"{model_type}_nside16/{model_type}_*", f"{model_type}_training_fixed/{model_type}_*", f"{model_type}_training/{model_type}_*"]
        else:
            patterns = [f"{model_type}/{model_type}_*", f"{model_type}_nside16/{model_type}_*", f"{model_type}_training/{model_type}_*"]

        latest_dir = None
        for pattern in patterns:
            dirs = list(results_dir.glob(pattern))
            if dirs:
                latest_dir = sorted(dirs)[-1]
                break

        if not latest_dir:
            print(f"Warning: No results found for {model_type}")
            continue
        eval_dir = latest_dir / 'evaluation'

        if not eval_dir.exists():
            print(f"Warning: No evaluation found for {model_type} at {eval_dir}")
            continue

        pred_file = eval_dir / 'predictions.npz'
        skymap_file = eval_dir / 'skymap_metrics.npz'

        if model_type == 'probmap' and skymap_file.exists():
            skymap_data = np.load(skymap_file)
            results[model_type] = {
                'angular_errors': skymap_data['angular_errors'],
                'searched_areas': skymap_data['searched_areas'],
                'credible_area_50': skymap_data['credible_area_50'],
                'credible_area_90': skymap_data['credible_area_90'],
                'eval_dir': eval_dir,
                'predictions': None,
                'targets': None,
            }
        elif pred_file.exists():
            data = np.load(pred_file)
            results[model_type] = {
                'predictions': data['predictions'],
                'targets': data['targets'],
                'angular_errors': data['angular_errors'],
                'eval_dir': eval_dir,
            }

            if model_type == 'classifier' and 'accuracy' in data:
                results[model_type]['accuracy'] = float(data['accuracy'])
                results[model_type]['predicted_pixels'] = data['predicted_pixels']
                results[model_type]['target_pixels'] = data['target_pixels']
            elif model_type == 'multitask':
                if 'tau_pred' in data:
                    results[model_type]['tau_pred'] = data['tau_pred']
                    results[model_type]['tau_target'] = data['tau_target']
                    results[model_type]['tau_errors'] = data['tau_errors']

    return results

def compute_statistics(results):
    stats = {}

    for model_type, data in results.items():
        errors = data['angular_errors']

        stats[model_type] = {
            'median': np.median(errors),
            'mean': np.mean(errors),
            'std': np.std(errors),
            'p10': np.percentile(errors, 10),
            'p25': np.percentile(errors, 25),
            'p75': np.percentile(errors, 75),
            'p90': np.percentile(errors, 90),
            'p95': np.percentile(errors, 95),
            'frac_lt_5': (errors < 5).mean(),
            'frac_lt_10': (errors < 10).mean(),
            'frac_lt_20': (errors < 20).mean(),
        }

        if model_type == 'classifier' and 'accuracy' in data:
            stats[model_type]['pixel_accuracy'] = data['accuracy']

        if model_type == 'multitask' and 'tau_errors' in data:
            tau_errors = data['tau_errors']
            stats[model_type]['tau_mae_ms'] = np.mean(np.abs(tau_errors)) * 1000
            stats[model_type]['tau_mae_HL_ms'] = np.mean(np.abs(tau_errors[:, 0])) * 1000
            stats[model_type]['tau_mae_HV_ms'] = np.mean(np.abs(tau_errors[:, 1])) * 1000

        if model_type == 'probmap' and 'searched_areas' in data:
            stats[model_type]['searched_area_median'] = np.median(data['searched_areas'])
            stats[model_type]['searched_area_mean'] = np.mean(data['searched_areas'])
            stats[model_type]['credible_50_median'] = np.median(data['credible_area_50'])
            stats[model_type]['credible_90_median'] = np.median(data['credible_area_90'])

    return stats

def plot_error_comparison(results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    model_names = {
        'regressor': 'Regressor\n(Unit vector prediction)',
        'classifier': 'Classifier\n(HEALPix classification)',
        'multitask': 'MultiTask\n(Sky + time delays)',
        'probmap': 'ProbabilityMap\n(Full distribution)',
    }

    colors = {
        'regressor': 'steelblue',
        'classifier': 'forestgreen',
        'multitask': 'darkorange',
        'probmap': 'crimson',
    }

    for idx, (model_type, data) in enumerate(results.items()):
        ax = axes[idx]
        errors = data['angular_errors']

        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color=colors[model_type])

        median = np.median(errors)
        mean = np.mean(errors)
        ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}°')
        ax.axvline(mean, color='blue', linestyle=':', linewidth=2, label=f'Mean: {mean:.2f}°')

        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(model_names[model_type], fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        frac_lt_5 = (errors < 5).mean() * 100
        frac_lt_10 = (errors < 10).mean() * 100
        textstr = f'< 5°: {frac_lt_5:.1f}%\n< 10°: {frac_lt_10:.1f}%'
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Angular Error Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")

def plot_cumulative_comparison(results, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    model_names = {
        'regressor': 'Regressor',
        'classifier': 'Classifier',
        'multitask': 'MultiTask',
        'probmap': 'ProbabilityMap',
    }

    colors = {
        'regressor': 'steelblue',
        'classifier': 'forestgreen',
        'multitask': 'darkorange',
        'probmap': 'crimson',
    }

    for model_type, data in results.items():
        if model_type == 'probmap':
            continue

        errors = np.sort(data['angular_errors'])
        cdf = np.arange(1, len(errors) + 1) / len(errors)

        median = np.median(errors)
        ax.plot(errors, cdf, linewidth=2.5, label=f'{model_names[model_type]} (median: {median:.2f}°)',
                color=colors[model_type])

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(10, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Angular Error (degrees)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title('Cumulative Error Distribution - Model Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")

def plot_percentile_comparison(stats, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    model_names = ['Regressor', 'Classifier', 'MultiTask', 'ProbabilityMap']
    model_keys = ['regressor', 'classifier', 'multitask', 'probmap']

    metrics = ['median', 'mean', 'p90']
    metric_names = ['Median', 'Mean', '90th Percentile']
    colors = ['steelblue', 'forestgreen', 'crimson']

    x = np.arange(len(model_names))
    width = 0.25

    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [stats[key][metric] for key in model_keys]
        ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.8)

        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 2, f'{v:.1f}°', ha='center', fontsize=10)

    ax.set_xlabel('Model Type', fontsize=14)
    ax.set_ylabel('Angular Error (degrees)', fontsize=14)
    ax.set_title('Model Comparison - Key Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")

def print_comparison_table(stats):
    print("\n" + "="*100)
    print("MODEL COMPARISON - ANGULAR ERROR STATISTICS")
    print("="*100)
    print(f"{'Model':<15} {'Median':<10} {'Mean':<10} {'P90':<10} {'<5°':<10} {'<10°':<10} {'<20°':<10}")
    print("-"*100)

    model_names = {
        'regressor': 'Regressor',
        'classifier': 'Classifier',
        'multitask': 'MultiTask',
        'probmap': 'ProbabilityMap',
    }

    for model_type in ['regressor', 'classifier', 'multitask', 'probmap']:
        if model_type not in stats:
            continue
        s = stats[model_type]
        print(f"{model_names[model_type]:<15} "
              f"{s['median']:>8.2f}°  "
              f"{s['mean']:>8.2f}°  "
              f"{s['p90']:>8.2f}°  "
              f"{s['frac_lt_5']*100:>8.1f}%  "
              f"{s['frac_lt_10']*100:>8.1f}%  "
              f"{s['frac_lt_20']*100:>8.1f}%")

    print("="*100)

    print("\nADDITIONAL METRICS:")
    print("-"*100)

    if 'classifier' in stats and 'pixel_accuracy' in stats['classifier']:
        print(f"Classifier - Pixel Accuracy: {stats['classifier']['pixel_accuracy']*100:.2f}%")

    if 'multitask' in stats and 'tau_mae_ms' in stats['multitask']:
        s = stats['multitask']
        print(f"MultiTask - Time Delay MAE: {s['tau_mae_ms']:.3f} ms (HL: {s['tau_mae_HL_ms']:.3f} ms, HV: {s['tau_mae_HV_ms']:.3f} ms)")

    if 'probmap' in stats and 'searched_area_median' in stats['probmap']:
        s = stats['probmap']
        print(f"ProbabilityMap - Searched Area: {s['searched_area_median']:.1f} deg² (median)")
        print(f"ProbabilityMap - 50% Credible Region: {s['credible_50_median']:.1f} deg²")
        print(f"ProbabilityMap - 90% Credible Region: {s['credible_90_median']:.1f} deg²")

    print("="*100)

def main():
    parser = argparse.ArgumentParser(description='Compare all Romain models')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory containing all model training runs')
    parser.add_argument('--output-dir', type=str, default='results/model_comparison', help='Output directory for comparison results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        print("ERROR: No results found!")
        sys.exit(1)

    stats = compute_statistics(results)
    print_comparison_table(stats)
    plot_error_comparison(results, output_dir / 'error_distributions.pdf')
    plot_cumulative_comparison(results, output_dir / 'cumulative_comparison.pdf')
    plot_percentile_comparison(stats, output_dir / 'percentile_comparison.pdf')

    stats_file = output_dir / 'comparison_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("ROMAIN MODEL COMPARISON - DETAILED STATISTICS\n")
        f.write("="*100 + "\n\n")

        for model_type, s in stats.items():
            f.write(f"\n{model_type.upper()}:\n")
            f.write("-"*50 + "\n")
            for key, value in s.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

    print(f"  Saved: {stats_file.name}")

if __name__ == '__main__':
    main()