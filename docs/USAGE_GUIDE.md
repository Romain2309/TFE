# Comprehensive Usage Guide

This guide provides detailed instructions for using the gravitational wave sky localization framework.

---

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Training Models](#training-models)
3. [Evaluation](#evaluation)
4. [Model Comparison](#model-comparison)
5. [Analysis and Visualization](#analysis-and-visualization)
6. [Advanced Usage](#advanced-usage)

---

## Data Preparation

### Input Data Format

The framework expects HDF5 files with the following structure:

```python
import h5py

# Required datasets
with h5py.File('dataset.h5', 'r') as f:
    strain_data = f['strain_data'][:]    # Shape: (N, 3, 2048)
    sky_positions = f['sky_positions'][:] # Shape: (N, 3)
    # Optional
    source_params = f['source_params'][:]  # Shape: (N, M)
```

**Field descriptions**:
- `strain_data`: Time-domain detector data
  - Dimension 0: Event index
  - Dimension 1: Detector (0=H1, 1=L1, 2=V1)
  - Dimension 2: Time samples (2048 @ 4096 Hz = 0.5 seconds)
- `sky_positions`: True source positions as unit vectors (x, y, z)
- `source_params`: Optional astrophysical parameters (masses, distance, etc.)

### Preprocessing

Preprocess raw gravitational wave data:

```bash
python scripts/preprocess_dataset.py \
    --input /data/raw/injections.hdf5 \
    --output /data/processed/benchmark_nside16.h5 \
    --nside 16 \
    --duration 0.5 \
    --sample-rate 4096
```

**Parameters**:
- `--input`: Path to raw data file
- `--output`: Path for processed output
- `--nside`: HEALPix resolution (8, 16, or 32)
- `--duration`: Signal duration in seconds
- `--sample-rate`: Sampling frequency in Hz

---

## Training Models

### Basic Training

Train a single model:

```bash
python scripts/train.py \
    --model regressor \
    --dataset /data/benchmark_small_nside16.h5 \
    --nside 16 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --output results/regressor_training
```

### Model Types

#### 1. Regressor
Direct regression to 3D unit vector:
```bash
python scripts/train.py --model regressor --dataset data.h5 --nside 16
```

#### 2. Classifier
Classification over HEALPix pixels:
```bash
python scripts/train.py --model classifier --dataset data.h5 --nside 16
```

#### 3. Multi-Task
Combined regression + classification:
```bash
python scripts/train.py --model multitask --dataset data.h5 --nside 16
```

#### 4. Probability Map
Full-sky probability distribution:
```bash
python scripts/train.py --model probmap --dataset data.h5 --nside 16
```

### Training Configuration

#### GPU Selection
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model regressor ...

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --model regressor ...
```

#### Hyperparameter Tuning
```bash
python scripts/train.py \
    --model multitask \
    --dataset data.h5 \
    --nside 16 \
    --lr 5e-4 \              # Learning rate
    --batch-size 128 \        # Batch size
    --weight-decay 1e-5 \     # L2 regularization
    --dropout 0.3 \           # Dropout rate
    --epochs 150              # Maximum epochs
```

#### Early Stopping
```bash
python scripts/train.py \
    --model regressor \
    --dataset data.h5 \
    --early-stopping \
    --patience 15             # Stop if no improvement for 15 epochs
```

### Monitoring Training

Training progress is logged to console and saved to:
- `results/model_name/training.log`: Text log
- `results/model_name/training_results_metrics.json`: Metrics history
- `results/model_name/best_model.pth`: Best checkpoint

**Example output**:
```
Epoch 10/100
Train Loss: 0.0875 | Val Loss: 0.0912
Val Metrics: Median=2.34° Mean=3.12° P90=5.67°
```

---

## Evaluation

### Basic Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --model-path results/regressor_training/best_model.pth \
    --model-type regressor \
    --dataset /data/test.h5 \
    --nside 16 \
    --output results/regressor_evaluation
```

**Outputs**:
- `evaluation/metrics.json`: Quantitative metrics
- `evaluation/error_histogram.pdf`: Error distribution
- `evaluation/sky_comparison.pdf`: Predicted vs true positions
- `evaluation/cumulative_distribution.pdf`: Cumulative error plot
- `evaluation/error_vs_position.pdf`: Sky-dependent performance

### Probability Map Evaluation

For models that output full probability maps:

```bash
python scripts/evaluate_with_skymaps.py \
    --model-path results/probmap_training/best_model.pth \
    --model-type probmap \
    --dataset /data/test.h5 \
    --nside 16 \
    --output results/probmap_evaluation \
    --n-skymaps 10            # Number of example sky maps to save
```

**Additional outputs**:
- `evaluation/searched_area_histogram.pdf`: Area at credible levels
- `evaluation/skymap_000.pdf` through `skymap_009.pdf`: Example probability maps
- `evaluation/credible_regions_50.pdf`: 50% credible regions
- `evaluation/credible_regions_90.pdf`: 90% credible regions

### Metrics Explained

#### Angular Error
Distance between predicted and true positions on the sphere:
```
θ = arccos(v_pred · v_true)
```
Reported in degrees.

#### Percentiles
- **Median (50th percentile)**: Typical performance
- **Mean**: Average over all test events
- **90th percentile**: Upper bound on 90% of errors

#### Searched Area
For probability maps: sky area that must be searched to find the true source with confidence C%:
```
Area(C%) = minimum sky area containing C% of probability mass
```

---

## Model Comparison

### Compare Multiple Models

```bash
python scripts/compare_models.py \
    --model-dir results/ \
    --dataset /data/test.h5 \
    --nside 16 \
    --output results/model_comparison
```

This finds all trained models in `results/` and generates:
- `error_distributions.pdf`: Error histograms for all models
- `cumulative_comparison.pdf`: Cumulative distribution functions
- `percentile_comparison.pdf`: Performance at different percentiles

### Specify Models Manually

```bash
python scripts/compare_models.py \
    --models \
        results/regressor_training/best_model.pth \
        results/classifier_training/best_model.pth \
        results/multitask_training/best_model.pth \
    --model-types regressor classifier multitask \
    --dataset /data/test.h5 \
    --nside 16 \
    --output comparison_results/
```

---

## Analysis and Visualization

### Comprehensive Analysis Plots

Generate all analysis plots:

```bash
python scripts/create_analysis_plots.py
```

**Generated plots**:

1. **Model Comparison by Dataset/Nside** (9 plots)
   - `{dataset}_nside{N}_model_comparison.pdf`
   - Compares all 4 models on same configuration
   - Shows median, mean, and 90th percentile errors

2. **Nside Impact Analysis** (3 plots)
   - `{dataset}_nside_impact.pdf`
   - Shows how performance varies with HEALPix resolution
   - One plot per dataset, all models on same axes

3. **Dataset Comparison** (1 plot)
   - `dataset_comparison_nside16.pdf`
   - Compares 3 datasets at fixed resolution
   - Separate subplot for each model

4. **Performance Heatmaps** (4 plots)
   - `{model}_performance_heatmap.pdf`
   - 2D heatmap: datasets × nside values
   - Color indicates median error

### Custom Plots

Create custom plots using the Python API:

```python
import matplotlib.pyplot as plt
from gw_skylocalization.evaluation.metrics import calculate_angular_errors

# Load results
predictions = np.load('predictions.npy')
targets = np.load('targets.npy')

# Calculate errors
errors = calculate_angular_errors(predictions, targets)

# Create custom plot
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel(r'Angular Error ($^\circ$)')
plt.ylabel('Counts')
plt.title('Custom Error Distribution')
plt.savefig('custom_plot.pdf')
```

---

## Advanced Usage

### Training on Multiple Resolutions

Train the same model at different nside values:

```bash
for nside in 8 16 32; do
    python scripts/train.py \
        --model multitask \
        --dataset /data/benchmark_small_nside${nside}.h5 \
        --nside ${nside} \
        --output results/multitask_nside${nside}
done
```

### Batch Processing

Process multiple datasets:

```bash
#!/bin/bash

datasets=("benchmark_small" "benchmark_same_small" "advanced")
models=("regressor" "classifier" "multitask" "probmap")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python scripts/train.py \
            --model ${model} \
            --dataset /data/${dataset}_nside16.h5 \
            --nside 16 \
            --output results/${dataset}/${model}
    done
done
```

### Fine-Tuning Pre-Trained Models

Load a checkpoint and continue training:

```bash
python scripts/train.py \
    --model regressor \
    --dataset /data/advanced_nside16.h5 \
    --nside 16 \
    --checkpoint results/benchmark_small/regressor/best_model.pth \
    --lr 1e-4 \               # Lower learning rate for fine-tuning
    --epochs 50 \
    --output results/advanced/regressor_finetuned
```

### Using the Python API

#### Training

```python
from gw_skylocalization.training.trainer import Trainer
from gw_skylocalization.models.networks import create_model
from gw_skylocalization.data.loaders import create_dataloaders

# Create model
model = create_model('multitask', nside=16)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    'data.h5',
    batch_size=64,
    num_workers=4
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-3,
    device='cuda'
)

# Train
trainer.train(epochs=100, output_dir='results/')
```

#### Evaluation

```python
from gw_skylocalization.evaluation.metrics import evaluate_model
import torch

# Load model
model = torch.load('results/best_model.pth')
model.eval()

# Evaluate
metrics = evaluate_model(
    model,
    test_loader,
    device='cuda',
    nside=16
)

print(f"Median error: {metrics['median_error']:.2f}°")
print(f"Mean error: {metrics['mean_error']:.2f}°")
```

### Exporting Models

#### Export to ONNX

```python
import torch

model = torch.load('best_model.pth')
model.eval()

dummy_input = torch.randn(1, 3, 2048)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['strain_data'],
    output_names=['sky_position'],
    dynamic_axes={'strain_data': {0: 'batch_size'}}
)
```

#### TorchScript

```python
model = torch.load('best_model.pth')
model.eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 2048))
traced_model.save('model_traced.pt')
```

---

## Tips and Best Practices

### Data Augmentation

Add noise augmentation during training:
```python
# In training loop
strain_data = strain_data + torch.randn_like(strain_data) * noise_std
```

### Learning Rate Scheduling

Use cosine annealing:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

### Mixed Precision Training

Speed up training with automatic mixed precision:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in train_loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Monitoring GPU Usage

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/train.py --batch-size 32  # Instead of 64
```

### Issue: Training is Slow

**Solutions**:
1. Enable mixed precision (see above)
2. Increase number of data loading workers:
   ```bash
   python scripts/train.py --num-workers 8
   ```
3. Use GPU if available

### Issue: Poor Performance

**Solutions**:
1. Check data normalization
2. Reduce learning rate
3. Increase model capacity
4. Add regularization (dropout, weight decay)
5. Train for more epochs

### Issue: NaN Losses

**Solutions**:
1. Reduce learning rate
2. Enable gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
3. Check for invalid data (inf, nan)

---

For more help, see README.md or contact the author.
