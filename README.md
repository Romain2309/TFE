# Gravitational Wave Sky Localization using Deep Learning

Master's Thesis Project - Deep learning models for gravitational wave source localization from binary black hole merger signals.

## Overview

This project implements four neural network architectures to localize gravitational wave sources on the sky using strain data from LIGO-Virgo detectors (H1, L1, V1). The models predict the sky position (Right Ascension and Declination) of binary black hole merger events.

### Model Architectures

- **Regressor**: Direct unit vector regression to predict sky position as a 3D unit vector
- **Classifier**: HEALPix pixel classification - discretizes the sky into pixels and classifies
- **Multi-Task**: Combined regression and classification with time delay prediction
- **Probability Map**: Outputs full-sky probability distribution as a HEALPix map

All models use a hybrid CNN-Transformer backbone architecture:
- **CNN backbone** (ResNet-based): Extracts features from strain time series
- **Transformer encoder**: Captures temporal dependencies and cross-detector correlations
- **Task-specific heads**: Different output layers for each architecture

### Dataset Types

Three benchmark datasets with different complexity levels:

1. **Advanced** (~50K events): Fully randomized parameters (duration, chirp, frequency evolution, amplitude, sky position)
2. **Benchmark Small** (~30K events): 4 fixed signal configurations, randomized sky positions
3. **Benchmark Same Small** (~30K events): 1 fixed signal configuration, Gaussian noise, randomized sky positions

Each dataset is preprocessed at 3 HEALPix resolutions: **nside=8** (768 pixels), **nside=16** (3072 pixels), **nside=32** (12288 pixels)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gw-skylocalization.git
cd gw-skylocalization
```

### 2. Install dependencies

**PyTorch with CUDA** (recommended for GPU training):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Core dependencies**:
```bash
pip install numpy scipy h5py healpy matplotlib pandas tqdm
```

**For data generation** (requires GWpyxel):
```bash
pip install gwpy
pip install git+https://github.com/AndyC80297/GWpyxel.git
```

### 3. Install this package

```bash
pip install -e .
```

## Repository Structure

```
gw-skylocalization/
├── data_generation/           # Scripts to generate datasets from scratch
│   ├── generate_parameters.py # Generate event parameters (RA/DEC, chirp params)
│   ├── simulate_event.py      # Simulate GW strain data using GWpyxel
│   └── generate_dataset.py    # Orchestrate full dataset generation
├── gw_skylocalization/        # Core Python package
│   ├── data/                  # Data loaders and file readers
│   ├── models/                # Neural network architectures
│   │   ├── backbones.py       # CNN + Transformer backbones
│   │   ├── heads.py           # Task-specific output heads
│   │   ├── networks.py        # Full model definitions
│   │   └── losses.py          # Custom loss functions
│   ├── training/              # Training infrastructure
│   │   ├── trainer.py         # Training loop
│   │   ├── config.py          # Training configurations
│   │   └── callbacks.py       # Logging, checkpointing
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # Angular error, searched area, etc.
│   └── utils/                 # Utility functions
│       ├── coordinates.py     # RA/DEC <-> HEALPix conversions
│       ├── physics.py         # GW signal processing
│       └── io.py              # File I/O helpers
├── scripts/                   # Training and evaluation scripts
│   ├── train.py               # Train models
│   ├── evaluate.py            # Evaluate models
│   ├── preprocess_dataset.py  # Convert .npy to HDF5
│   ├── compare_models.py      # Compare all model results
│   └── create_analysis_plots.py # Generate analysis figures
└── results/                   # Trained models and evaluation results
    ├── advanced/
    ├── benchmark_small/
    └── benchmark_same_small/
```

## Generating Datasets from Scratch

Since the preprocessed datasets are very large (>3TB uncompressed), they are not included in this repository. Follow these steps to regenerate them:

### Prerequisites

You must have **GWpyxel** installed for gravitational wave simulation:

```bash
pip install gwpy
pip install git+https://github.com/AndyC80297/GWpyxel.git
```

### Step 1: Generate Raw Strain Data

Each dataset type has different generation parameters:

#### Advanced Dataset (~50,000 events)

```bash
cd data_generation

# Generate event parameters (RA, DEC, chirp duration, frequencies, etc.)
python generate_parameters.py \
    --dataset advanced \
    --n_events 50000 \
    --output /data/advanced/parameters.csv

# Simulate all events (this will take several hours/days)
python generate_dataset.py \
    --dataset advanced \
    --n_events 50000 \
    --output_dir /data/advanced \
    --n_jobs 4  # Parallel jobs
```

**Output**: `/data/advanced/` will contain ~150,000 `.npy` files (3 detectors × 50K events)

#### Benchmark Small Dataset (~30,000 events)

```bash
python generate_parameters.py \
    --dataset benchmark_small \
    --n_events 30000 \
    --output /data/benchmark_small/parameters.csv

python generate_dataset.py \
    --dataset benchmark_small \
    --n_events 30000 \
    --output_dir /data/benchmark_small \
    --n_jobs 4
```

**Output**: `/data/benchmark_small/` will contain ~90,000 `.npy` files

#### Benchmark Same Small Dataset (~30,000 events)

```bash
python generate_parameters.py \
    --dataset benchmark_same_small \
    --n_events 30000 \
    --output /data/benchmark_same_small/parameters.csv

python generate_dataset.py \
    --dataset benchmark_same_small \
    --n_events 30000 \
    --output_dir /data/benchmark_same_small \
    --n_jobs 4
```

**Output**: `/data/benchmark_same_small/` will contain ~90,000 `.npy` files

**Estimated generation time**:
- Advanced: ~48-72 hours (highly variable parameters)
- Benchmark Small: ~24-36 hours
- Benchmark Same Small: ~24-36 hours

### Step 2: Preprocess to HDF5

The raw `.npy` files are slow to load. Convert them to HDF5 format with bandpass filtering and normalization:

#### Preprocess for each HEALPix resolution

You need to preprocess each dataset at 3 different HEALPix resolutions (nside=8, 16, 32):

**Advanced Dataset:**
```bash
cd scripts

# nside=8 (768 sky pixels)
python preprocess_dataset.py \
    --input-dir /data/advanced \
    --output-dir /data/preprocessed/advanced_nside8 \
    --healpix-nside 8 \
    --normalize

# nside=16 (3072 sky pixels)
python preprocess_dataset.py \
    --input-dir /data/advanced \
    --output-dir /data/preprocessed/advanced_nside16 \
    --healpix-nside 16 \
    --normalize

# nside=32 (12288 sky pixels)
python preprocess_dataset.py \
    --input-dir /data/advanced \
    --output-dir /data/preprocessed/advanced_nside32 \
    --healpix-nside 32 \
    --normalize
```

**Benchmark Small:**
```bash
python preprocess_dataset.py \
    --input-dir /data/benchmark_small \
    --output-dir /data/preprocessed/benchmark_small_nside8 \
    --healpix-nside 8 \
    --normalize

python preprocess_dataset.py \
    --input-dir /data/benchmark_small \
    --output-dir /data/preprocessed/benchmark_small_nside16 \
    --healpix-nside 16 \
    --normalize

python preprocess_dataset.py \
    --input-dir /data/benchmark_small \
    --output-dir /data/preprocessed/benchmark_small_nside32 \
    --healpix-nside 32 \
    --normalize
```

**Benchmark Same Small:**
```bash
python preprocess_dataset.py \
    --input-dir /data/benchmark_same_small \
    --output-dir /data/preprocessed/benchmark_same_small_nside8 \
    --healpix-nside 8 \
    --normalize

python preprocess_dataset.py \
    --input-dir /data/benchmark_same_small \
    --output-dir /data/preprocessed/benchmark_same_small_nside16 \
    --healpix-nside 16 \
    --normalize

python preprocess_dataset.py \
    --input-dir /data/benchmark_same_small \
    --output-dir /data/preprocessed/benchmark_same_small_nside32 \
    --healpix-nside 32 \
    --normalize
```

**Output**: Each preprocessing creates a `preprocessed_data.h5` file (~700MB-1.2GB each)

**Preprocessing options**:
- `--normalize`: Normalize strain data (recommended, set to False for benchmark_same_small nside8)
- `--bandpass-low 30 --bandpass-high 1024`: Frequency range in Hz
- `--target-length 2048`: Time series length

### Step 3: File Organization

After preprocessing, organize your data directory like this:

```
/data/
├── preprocessed/
│   ├── advanced_nside8/preprocessed_data.h5
│   ├── advanced_nside16/preprocessed_data.h5
│   ├── advanced_nside32/preprocessed_data.h5
│   ├── benchmark_small_nside8/preprocessed_data.h5
│   ├── benchmark_small_nside16/preprocessed_data.h5
│   ├── benchmark_small_nside32/preprocessed_data.h5
│   ├── benchmark_same_small_nside8/preprocessed_data.h5
│   ├── benchmark_same_small_nside16/preprocessed_data.h5
│   └── benchmark_same_small_nside32/preprocessed_data.h5
└── [raw .npy files can be archived or deleted after preprocessing]
```

**Disk space requirements**:
- Raw .npy files: ~3TB total (can be deleted after preprocessing)
- Preprocessed .h5 files: ~7-8GB total

## Training Models

### Quick Start

Train a single model:

```bash
cd scripts

python train.py \
    --model regressor \
    --data-path /data/preprocessed/benchmark_small_nside16/preprocessed_data.h5 \
    --nside 16 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir results/benchmark_small_nside16/regressor
```

### Train All Model Configurations

To reproduce all 36 trained models (4 architectures × 3 datasets × 3 nsides):

```bash
# Example: Train all 4 models on benchmark_small at nside=16
for model in regressor classifier multitask probmap; do
    python train.py \
        --model $model \
        --data-path /data/preprocessed/benchmark_small_nside16/preprocessed_data.h5 \
        --nside 16 \
        --epochs 100 \
        --batch-size 32 \
        --output-dir results/benchmark_small_nside16/$model
done
```

**Training parameters**:
- `--model`: Choose from `regressor`, `classifier`, `multitask`, `probmap`
- `--nside`: HEALPix resolution (8, 16, or 32)
- `--batch-size`: Depends on GPU memory (32 for nside=8/16, 16 for nside=32)
- `--lr`: Learning rate (default 1e-4)
- `--weight-decay`: L2 regularization (default 1e-5)
- `--epochs`: Number of training epochs (100-200 recommended)

**Training time** (on NVIDIA A100):
- Regressor/Classifier: ~2-4 hours per model
- MultiTask: ~4-6 hours
- ProbabilityMap: ~6-10 hours (largest output)

## Evaluating Models

### Evaluate a single model

```bash
python evaluate.py \
    --model regressor \
    --checkpoint results/benchmark_small_nside16/regressor/best_model.pt \
    --data-path /data/preprocessed/benchmark_small_nside16/preprocessed_data.h5 \
    --nside 16 \
    --output-dir results/benchmark_small_nside16/regressor/evaluation
```

### Compare all models

Generate comparison plots and statistics:

```bash
python compare_models.py \
    --results-dir results/benchmark_small_nside16 \
    --output-dir results/benchmark_small_nside16/comparison
```

**Metrics computed**:
- **Angular error**: Great circle distance between predicted and true position
- **Searched area**: Sky area needed to find true source (for ProbabilityMap)
- **Credible regions**: 50% and 90% credible areas
- **Pixel accuracy**: Classification accuracy (for Classifier)
- **Time delay error**: Prediction accuracy for detector time delays (for MultiTask)

## Results

The `results/` directory contains all 36 trained models with their evaluation metrics:

```
results/
├── advanced/
│   ├── nside8/  {regressor, classifier, multitask, probmap}
│   ├── nside16/ {regressor, classifier, multitask, probmap}
│   └── nside32/ {regressor, classifier, multitask, probmap}
├── benchmark_small/
│   ├── nside8/  {regressor, classifier, multitask, probmap}
│   ├── nside16/ {regressor, classifier, multitask, probmap}
│   └── nside32/ {regressor, classifier, multitask, probmap}
└── benchmark_same_small/
    ├── nside8/  {regressor, classifier, multitask, probmap}
    ├── nside16/ {regressor, classifier, multitask, probmap}
    └── nside32/ {regressor, classifier, multitask, probmap}
```

Each model directory contains:
- `best_model.pt`: Trained model checkpoint
- `config.json`: Training configuration
- `training_results_metrics.json`: Training metrics
- `evaluation/`: Evaluation results and plots

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{gwskylocalization2026,
  title={Deep Learning for Gravitational Wave Sky Localization},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- LIGO-Virgo Collaboration for gravitational wave data
- GWpyxel library for gravitational wave simulation
- HEALPix for sky discretization scheme
