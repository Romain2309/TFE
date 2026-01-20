# Gravitational Wave Sky Localization using Deep Learning

## Overview

This project implements four neural network architectures to localize gravitational wave sources on the sky using generated strain data from LIGO-Virgo detectors (H1, L1, V1). The models predict the sky position (Right Ascension and Declination) on long bursts.

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
git clone https://github.com/Romain2309/TFE.git
cd TFE
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

Since the preprocessed datasets are large, they are not included in this repository. Follow these steps to regenerate them:

### Prerequisites

You must have **GWpyxel** installed for gravitational wave simulation:

```bash
pip install gwpy
pip install git+https://https://git.ligo.org/maxime.fays/gwpyxel.git
```

### Step 1: Generate Raw Strain Data

Generate event parameters and simulate GW strain data for each dataset type:

```bash
cd data_generation

# dataset in [advanced, benchmark_small, benchmark_same_small]
# n_events: 50000 for advanced, 30000 for others (the values used in this work, it can be others)

python generate_parameters.py \
    --dataset <dataset> \
    --n_events <n_events> \
    --output /data/<dataset>/parameters.csv

python generate_dataset.py \
    --dataset <dataset> \
    --n_events <n_events> \
    --output_dir /data/<dataset> \
    --n_jobs 4
```

**Output**: Each dataset directory will contain `.npy` files (3 detectors × n_events)

### Step 2: Preprocess to HDF5

Convert raw `.npy` files to HDF5 format with bandpass filtering and normalization:

```bash
cd scripts

# dataset in [advanced, benchmark_small, benchmark_same_small]
# healpix_nside in [8, 16, 32] (the values used in this work, it can be others)

python preprocess_dataset.py \
    --input-dir /data/<dataset> \
    --output-dir /data/preprocessed/<dataset>_nside<healpix_nside> \
    --healpix-nside <healpix_nside> \
    --normalize \
    --bandpass-low 30 \
    --bandpass-high 1024 \
    --target-length 2048
```

**Output**: Each preprocessing creates a `preprocessed_data.h5` file in the output directory

### Step 3: File Organization

After preprocessing, the expected directory structure is:

```
/data/preprocessed/
├── <dataset>_nside<healpix_nside>/preprocessed_data.h5
└── ... (9 total: 3 datasets × 3 nsides)
```

Raw `.npy` files can be archived or deleted after preprocessing.

## Training Models

```bash
cd scripts

# model in [regressor, classifier, multitask, probmap]
# dataset in [advanced, benchmark_small, benchmark_same_small]
# healpix_nside in [8, 16, 32] (the values used in this work, it can be others)

python train.py \
    --model <model> \
    --data-path /data/preprocessed/<dataset>_nside<healpix_nside>/preprocessed_data.h5 \
    --nside <healpix_nside> \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --output-dir results/<dataset>_nside<healpix_nside>/<model>
```

To train all 36 models (4 architectures × 3 datasets × 3 nsides), iterate over the parameters.

## Evaluating Models

### Evaluate a model

```bash
python evaluate.py \
    --model <model> \
    --checkpoint results/<dataset>_nside<healpix_nside>/<model>/best_model.pt \
    --data-path /data/preprocessed/<dataset>_nside<healpix_nside>/preprocessed_data.h5 \
    --nside <healpix_nside> \
    --output-dir results/<dataset>_nside<healpix_nside>/<model>/evaluation
```

### Compare models

```bash
python compare_models.py \
    --results-dir results/<dataset>_nside<healpix_nside> \
    --output-dir results/<dataset>_nside<healpix_nside>/comparison
```

**Metrics**: Angular error, searched area, credible regions, pixel accuracy, time delay error

## Results

The `results/` directory contains all 36 trained models:

```
results/<dataset>_nside<healpix_nside>/<model>/
├── best_model.pt
├── config.json
├── training_results_metrics.json
└── evaluation/
```

**Structure**: 4 model architectures × 3 datasets × 3 HEALPix resolutions = 36 configurations
