#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

def collect_all_metrics(results_dir):
    results_dir = Path(results_dir)
    all_metrics = {}

    datasets = ['benchmark_small', 'benchmark_same_small', 'advanced']
    nsides = [8, 16, 32]
    models = ['regressor', 'classifier', 'multitask', 'probmap']

    for dataset in datasets:
        all_metrics[dataset] = {}
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            continue

        for nside in nsides:
            all_metrics[dataset][nside] = {}
            nside_dir = dataset_dir / f'nside{nside}'
            if not nside_dir.exists():
                continue

            for model in models:
                model_dir = nside_dir / f'{model}_training'
                if not model_dir.exists():
                    continue

                training_runs = [d for d in model_dir.iterdir() if d.is_dir()]
                if not training_runs:
                    continue

                latest_run = sorted(training_runs)[-1]
                metrics_file = latest_run / 'training_results_metrics.json'

                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        all_metrics[dataset][nside][model] = data['final_metrics']

    return all_metrics

def plot_model_comparison_by_dataset_nside(all_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = ['benchmark_small', 'benchmark_same_small', 'advanced']
    dataset_labels = {
        'benchmark_small': r'Benchmark Small',
        'benchmark_same_small': r'Benchmark Same Small',
        'advanced': r'Advanced'
    }

    nsides = [8, 16, 32]
    models = ['regressor', 'classifier', 'multitask', 'probmap']
    model_labels = {
        'regressor': r'Regressor',
        'classifier': r'Classifier',
        'multitask': r'Multi-Task',
        'probmap': r'Prob. Map'
    }

    colors = {
        'regressor': '#1f77b4',
        'classifier': '#ff7f0e',
        'multitask': '#2ca02c',
        'probmap': '#d62728'
    }

    for dataset in datasets:
        if dataset not in all_metrics:
            continue

        for nside in nsides:
            if nside not in all_metrics[dataset]:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f'{dataset_labels[dataset]} - nside={nside}', y=1.02)

            metrics_to_plot = [
                ('val_median_error', r'Median Error ($^\circ$)'),
                ('val_mean_error', r'Mean Error ($^\circ$)'),
                ('val_p90_error', r'90th Percentile Error ($^\circ$)')
            ]

            for ax_idx, (metric_key, ylabel) in enumerate(metrics_to_plot):
                ax = axes[ax_idx]

                model_names = []
                values = []
                bar_colors = []

                for model in models:
                    if model in all_metrics[dataset][nside]:
                        metrics = all_metrics[dataset][nside][model]
                        model_names.append(model_labels[model])
                        values.append(metrics.get(metric_key, 0))
                        bar_colors.append(colors[model])

                if values:
                    x = np.arange(len(model_names))
                    bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.8)

                    ax.set_ylabel(ylabel)
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=0)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            output_file = output_dir / f'{dataset}_nside{nside}_model_comparison.pdf'
            plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            print(f'Created: {output_file}')

def plot_nside_impact(all_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = ['benchmark_small', 'benchmark_same_small', 'advanced']
    dataset_labels = {
        'benchmark_small': r'Benchmark Small',
        'benchmark_same_small': r'Benchmark Same Small',
        'advanced': r'Advanced'
    }

    models = ['regressor', 'classifier', 'multitask', 'probmap']
    model_labels = {
        'regressor': r'Regressor',
        'classifier': r'Classifier',
        'multitask': r'Multi-Task',
        'probmap': r'Prob. Map'
    }

    colors = {
        'regressor': '#1f77b4',
        'classifier': '#ff7f0e',
        'multitask': '#2ca02c',
        'probmap': '#d62728'
    }

    markers = {
        'regressor': 'o',
        'classifier': 's',
        'multitask': '^',
        'probmap': 'D'
    }

    nsides = [8, 16, 32]

    for dataset in datasets:
        if dataset not in all_metrics:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'{dataset_labels[dataset]} - Performance vs Resolution', y=1.02)

        metrics_to_plot = [
            ('val_median_error', r'Median Error ($^\circ$)'),
            ('val_mean_error', r'Mean Error ($^\circ$)'),
            ('val_p90_error', r'90th Percentile Error ($^\circ$)')
        ]

        for ax_idx, (metric_key, ylabel) in enumerate(metrics_to_plot):
            ax = axes[ax_idx]

            for model in models:
                nside_values = []
                error_values = []

                for nside in nsides:
                    if nside in all_metrics[dataset] and model in all_metrics[dataset][nside]:
                        metrics = all_metrics[dataset][nside][model]
                        nside_values.append(nside)
                        error_values.append(metrics.get(metric_key, 0))

                if nside_values:
                    ax.plot(nside_values, error_values,
                           marker=markers[model],
                           color=colors[model],
                           label=model_labels[model],
                           linewidth=2,
                           markersize=8)

            ax.set_xlabel(r'HEALPix Resolution (nside)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(nsides)
            ax.grid(alpha=0.3, linestyle='--')
            if ax_idx == 2:
                ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        output_file = output_dir / f'{dataset}_nside_impact.pdf'
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f'Created: {output_file}')

def plot_dataset_comparison(all_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = ['benchmark_small', 'benchmark_same_small', 'advanced']
    dataset_labels = {
        'benchmark_small': r'Benchmark Small',
        'benchmark_same_small': r'Benchmark Same Small',
        'advanced': r'Advanced'
    }

    models = ['regressor', 'classifier', 'multitask', 'probmap']
    model_labels = {
        'regressor': r'Regressor',
        'classifier': r'Classifier',
        'multitask': r'Multi-Task',
        'probmap': r'Prob. Map'
    }

    nside = 16
    colors = {
        'benchmark_small': '#1f77b4',
        'benchmark_same_small': '#ff7f0e',
        'advanced': '#2ca02c'
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Dataset Comparison - nside={nside}', y=0.995)
    axes = axes.flatten()

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        metric_names = [r'Median', r'Mean', r'90th \%ile']
        metric_keys = ['val_median_error', 'val_mean_error', 'val_p90_error']

        x = np.arange(len(metric_names))
        width = 0.25

        for dataset_idx, dataset in enumerate(datasets):
            if dataset in all_metrics and nside in all_metrics[dataset]:
                if model in all_metrics[dataset][nside]:
                    metrics = all_metrics[dataset][nside][model]
                    values = [metrics.get(key, 0) for key in metric_keys]

                    offset = (dataset_idx - 1) * width
                    bars = ax.bar(x + offset, values, width,
                                 label=dataset_labels[dataset],
                                 color=colors[dataset],
                                 alpha=0.8,
                                 edgecolor='black',
                                 linewidth=0.8)

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}',
                               ha='center', va='bottom', fontsize=7)

        ax.set_ylabel(r'Error ($^\circ$)')
        ax.set_title(model_labels[model])
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / f'dataset_comparison_nside{nside}.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Created: {output_file}')

def plot_overall_performance_heatmap(all_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = ['benchmark_small', 'benchmark_same_small', 'advanced']
    dataset_labels = {
        'benchmark_small': r'B. Small',
        'benchmark_same_small': r'B. Same Small',
        'advanced': r'Advanced'
    }

    nsides = [8, 16, 32]
    models = ['regressor', 'classifier', 'multitask', 'probmap']
    model_labels = {
        'regressor': r'Regressor',
        'classifier': r'Classifier',
        'multitask': r'Multi-Task',
        'probmap': r'Prob. Map'
    }

    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))

        data_matrix = np.zeros((len(datasets), len(nsides)))

        for i, dataset in enumerate(datasets):
            for j, nside in enumerate(nsides):
                if dataset in all_metrics and nside in all_metrics[dataset]:
                    if model in all_metrics[dataset][nside]:
                        metrics = all_metrics[dataset][nside][model]
                        data_matrix[i, j] = metrics.get('val_median_error', 0)
                    else:
                        data_matrix[i, j] = np.nan
                else:
                    data_matrix[i, j] = np.nan

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(nsides)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels([f'nside={n}' for n in nsides])
        ax.set_yticklabels([dataset_labels[d] for d in datasets])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'Median Error ($^\circ$)', rotation=270, labelpad=20)

        for i in range(len(datasets)):
            for j in range(len(nsides)):
                if not np.isnan(data_matrix[i, j]):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)

        ax.set_title(f'{model_labels[model]} - Median Error Heatmap')
        plt.tight_layout()

        output_file = output_dir / f'{model}_performance_heatmap.pdf'
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f'Created: {output_file}')

def main():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    output_dir = base_dir / 'Analysis_Plots'

    all_metrics = collect_all_metrics(results_dir)
    plot_model_comparison_by_dataset_nside(all_metrics, output_dir)
    plot_nside_impact(all_metrics, output_dir)
    plot_dataset_comparison(all_metrics, output_dir)
    plot_overall_performance_heatmap(all_metrics, output_dir)


if __name__ == '__main__':
    main()
